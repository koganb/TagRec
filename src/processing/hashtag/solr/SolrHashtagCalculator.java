package processing.hashtag.solr;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import com.google.common.primitives.Ints;

import common.Bookmark;
import common.SolrConnector;
import file.BookmarkReader;
import file.PredictionFileWriter;
import file.ResultSerializer;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SolrHashtagCalculator {

    private final static int LIMIT = 10;

    private final static Logger logger = LoggerFactory.getLogger(SolrHashtagCalculator.class);


    // Statics ----------------------------------------------------------------------------------------------------------------------

    public static String predictSample(String sampleDir, String solrCore, String solrUrl) {
        List<Set<String>> predictionValues = new ArrayList<Set<String>>();
        List<Set<String>> realValues = new ArrayList<Set<String>>();
        SolrConnector trainConnector = new SolrConnector(solrUrl, solrCore + "_train");
        SolrConnector testConnector = new SolrConnector(solrUrl, solrCore + "_test");

        Map<String, Set<String>> tweets = testConnector.getTweets();
        for (Map.Entry<String, Set<String>> tweet : tweets.entrySet()) {
            if (tweet.getValue().size() > 0) {
                Map<String, Double> map = trainConnector.getTopHashtagsForTweetText(tweet.getKey(), LIMIT);
                predictionValues.add(map.keySet());
                realValues.add(tweet.getValue());
                if (predictionValues.size() % 100 == 0) {
                    System.out.println(predictionValues.size() + " users done. Left ones: " + (tweets.size() - predictionValues.size()));
                }
            }
        }
        String suffix = "solrht";
        PredictionFileWriter.writeSimplePredictions(predictionValues, realValues, null, sampleDir + "/" + solrCore + "_" + suffix);
        return suffix;
    }

    public static Map<Integer, Map<Integer, Double>> getNormalizedHashtagPredictions(String sampleDir, String solrCore, String solrUrl, BookmarkReader reader, Integer trainHours) {
        SolrConnector trainConnector = new SolrConnector(solrUrl, solrCore + "_train");
        SolrConnector testConnector = new SolrConnector(solrUrl, solrCore + "_test");

        //hashtages by tweet similarity
        Map<Integer, Map<Integer, Double>> predictedHashTagByContent = new LinkedHashMap<>();

        //hashtag keys
        List<Set<String>> predictionHashTagIds = new ArrayList<>();

        //hashtags from tweet
        List<Set<String>> tweetHashtags = new ArrayList<>();


        List<String> userIdTweetId = new ArrayList<String>();

        List<Tweet> tweets = null;
        if (trainHours == null) {
            tweets = testConnector.getTweetObjects(true);
            logger.info("Get twitter test tweet data (size {})", tweets.size());
        } else {
            tweets = testConnector.getTrainTweetObjects(trainConnector, trainHours);
            logger.info("Get twitter train data (size {}) for last {} hours", tweets.size(), trainHours);
        }
        for (Tweet tweet : tweets)
            if (tweet.getHashtags().size() > 0) {

                //get to hashtags for current tweet
                Map<String, Double> topHashtagsForTweetText = trainConnector.getTopHashtagsForTweetText(tweet.getText(), 50);

                //normalize the document score
                Map<Integer, Double> normalizedTweetScoreResult = getNormalizedTopHashtagForTweet(topHashtagsForTweetText, reader);
                Set<String> topHashtagsIds = normalizedTweetScoreResult.keySet().stream().map(Object::toString).collect(Collectors.toSet());

                //get userId of the current tweet
                Integer userId = reader.getUserMap().get(tweet.getUserid());

                if (userId != null) {
                    predictedHashTagByContent.put(userId, normalizedTweetScoreResult);
                    predictionHashTagIds.add(topHashtagsIds);
                    if (predictedHashTagByContent.size() % 100 == 0) {
                        System.out.println(predictedHashTagByContent.size() + " users done. Left ones: " + (tweets.size() - predictedHashTagByContent.size()));
                    }

                    // add real tweets hashtags
                    tweetHashtags.add(
                            tweet.getHashtags().stream().
                                    map(t -> reader.getTagMap().get(t.toLowerCase())).
                                    filter(t -> Optional.ofNullable(t).isPresent()).
                                    map(Object::toString).
                                    collect(Collectors.toSet())
                    );

                    userIdTweetId.add(userId + "-" + reader.getResourceMap().get(tweet.getId()));
                }
            }

        //printHashtagPrediction(hashtagMaps, "./data/results/" + sampleDir + "/" + solrCore + "_cbpredictions.txt");
        ResultSerializer.serializePredictions(predictedHashTagByContent, "./data/results/" + sampleDir + "/" + solrCore + "_cbpredictions.ser");
        PredictionFileWriter.writeSimplePredictions(predictionHashTagIds, tweetHashtags, userIdTweetId, sampleDir + "/" + solrCore + "_solrht_normalized");
        return predictedHashTagByContent;
    }

    private static Map<Integer, Double> getNormalizedTopHashtagForTweet(Map<String, Double> topHashTags, BookmarkReader reader) {
        double denom = topHashTags.values().stream().collect(Collectors.summingDouble(Math::exp));
        return topHashTags.entrySet().stream().map(t ->

                //convert tweet to ID
                ImmutablePair.of(reader.getTagMap().get(t.getKey().toLowerCase()), t.getValue())).

                //filter if the ID do not exits
                        filter(t -> Optional.ofNullable(t.getKey()).isPresent()).

                //normalize tweets value
                        collect(Collectors.toMap(Pair::getKey, t -> Math.exp(t.getValue()) / denom));
    }


    public static String predictTrainSample(String sampleDir, String solrCore, String solrUrl, boolean hours, Integer recentTweetThreshold) {
        List<Set<String>> predictionValues = new ArrayList<Set<String>>();
        List<Set<String>> realValues = new ArrayList<Set<String>>();
        SolrConnector trainConnector = new SolrConnector(solrUrl, solrCore + "_train");
        SolrConnector testConnector = new SolrConnector(solrUrl, solrCore + "_test");

        String suffix = "";
        Map<String, Set<String>> userIDs = testConnector.getUserIDs();
        for (Map.Entry<String, Set<String>> user : userIDs.entrySet()) {
            if (user.getValue().size() > 0) {
                Map<String, Double> map = null;
                if (recentTweetThreshold == null) {
                    String id = trainConnector.getMostRecentTweetOfUser(user.getKey());
                    map = trainConnector.getTopHashtagsForTweetID(id, LIMIT);
                    suffix = "solrht_train";
                } else {
                    String text = null;
                    if (hours) {
                        text = trainConnector.getTweetTextOfLastHours(user.getKey(), recentTweetThreshold.intValue());
                        suffix = "solrht_train_" + recentTweetThreshold.intValue() + "hours";
                    } else {
                        text = trainConnector.getTweetTextOfRecentTweets(user.getKey(), recentTweetThreshold.intValue());
                        suffix = "solrht_train_" + recentTweetThreshold.intValue();
                    }
                    map = trainConnector.getTopHashtagsForTweetText(text, LIMIT);
                }
                predictionValues.add(map.keySet());
                realValues.add(user.getValue());
                if (predictionValues.size() % 100 == 0) {
                    System.out.println(predictionValues.size() + " users done. Left ones: " + (userIDs.size() - predictionValues.size()));
                }
            }
        }

        PredictionFileWriter.writeSimplePredictions(predictionValues, realValues, null, sampleDir + "/" + solrCore + "_" + suffix);
        return suffix;
    }

    public static void printHashtagPrediction(Map<Integer, Map<Integer, Double>> predictions, String filePath) {
        try {
            FileWriter writer = new FileWriter(new File(filePath));
            BufferedWriter bw = new BufferedWriter(writer);

            for (Map.Entry<Integer, Map<Integer, Double>> predEntry : predictions.entrySet()) {
                bw.write(predEntry.getKey() + "|");
                int i = 1;
                for (Map.Entry<Integer, Double> mapEntry : predEntry.getValue().entrySet()) {
                    bw.write(mapEntry.getKey() + ":" + mapEntry.getValue());
                    if (i++ < predEntry.getValue().size()) {
                        bw.write(";");
                    }
                }
                bw.write("\n");
            }

            bw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Map<Integer, Map<Integer, Double>> deSerializeHashtagPrediction(String filePath) {
        InputStream file = null;
        Map<Integer, Map<Integer, Double>> predictions = null;
        try {
            file = new FileInputStream(filePath);
            InputStream buffer = new BufferedInputStream(file);
            ObjectInput input = new ObjectInputStream(buffer);
            predictions = (Map<Integer, Map<Integer, Double>>) input.readObject();
            input.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return predictions;
    }

    public static Map<Integer, Map<Integer, Double>> readHashtagPrediction(String filePath) {
        Map<Integer, Map<Integer, Double>> hashtagMaps = new LinkedHashMap<Integer, Map<Integer, Double>>();
        try {
            //FileReader reader = new FileReader(new File(filePath));
            InputStreamReader reader = new InputStreamReader(new FileInputStream(new File(filePath)), "UTF8");
            BufferedReader br = new BufferedReader(reader);
            String line = null;
            while ((line = br.readLine()) != null) {
                Map<Integer, Double> tagMap = new LinkedHashMap<Integer, Double>();
                String[] parts = line.split("\\|");
                int userID = Integer.parseInt(parts[0]);
                if (parts.length > 1) {
                    String[] tags = parts[1].split(";");
                    for (String t : tags) {
                        String[] tParts = t.split(":");
                        if (tParts.length > 1 && !tParts[0].equals("null")) {
                            try {
                                tagMap.put(Integer.parseInt(tParts[0]), Double.parseDouble(tParts[1]));
                            } catch (Exception e) {
                                System.out.println("Parse Exception: " + tParts[0] + " " + tParts[1]);
                            }
                        }
                    }
                }
                hashtagMaps.put(userID, tagMap);
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return hashtagMaps;
    }
}
