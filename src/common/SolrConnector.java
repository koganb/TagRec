package common;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrRequest.METHOD;
import org.apache.solr.client.solrj.SolrServer;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.impl.HttpSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrDocumentList;
import org.apache.solr.common.params.MoreLikeThisParams;

import processing.hashtag.solr.Tweet;

public class SolrConnector {
	
	private SolrServer server;
	
	public SolrConnector(String solrUrl, String core) {
		this.server = new HttpSolrServer(solrUrl + "/solr/" + core);
	}
	
	@SuppressWarnings("unchecked")
	public Map<String, Set<String>> getTweets() {
		Map<String, Set<String>> tweets = new LinkedHashMap<String, Set<String>>();
		
		SolrQuery solrParams = new SolrQuery();
		solrParams.set("q", "*:*");
		solrParams.set("fl", "text,hashtags");
		solrParams.set("rows", Integer.MAX_VALUE);
		QueryResponse r = null;
		try {
			r = this.server.query(solrParams);
			SolrDocumentList docs = r.getResults();
			for (SolrDocument d : docs) {
				tweets.put((String) d.get("text"), new LinkedHashSet<String>((List<String>) d.get("hashtags")));
			}
		} catch (SolrServerException e) {
			e.printStackTrace();
		}
		
		return tweets;
	}
	
	@SuppressWarnings("unchecked")
	public List<Tweet> getTweetObjects(boolean ignoreRetweets) {
		List<Tweet> tweetObjects = new ArrayList<Tweet>();
		
		SolrQuery solrParams = new SolrQuery();
		if (!ignoreRetweets) {
			solrParams.set("q", "*:*");
		} else {
			solrParams.set("q", "-text:\"RT @*\"");
		}
		solrParams.set("rows", Integer.MAX_VALUE);
		QueryResponse r = null;
		try {
			r = this.server.query(solrParams);
			SolrDocumentList docs = r.getResults();
			for (SolrDocument d : docs) {
				tweetObjects.add(new Tweet((String) d.get("id"), (String) d.get("userid"), (String) d.get("text"), (String) d.get("timestamp"), new LinkedHashSet<String>((List<String>) d.get("hashtags"))));
			}
		} catch (SolrServerException e) {
			e.printStackTrace();
		}
		
		return tweetObjects;
	}
	
	@SuppressWarnings("unchecked")
	public List<Tweet> getTrainTweetObjects(SolrConnector trainConnector, int hours) {
		List<Tweet> tweetObjects = new ArrayList<Tweet>();
		
		SolrQuery solrParams = new SolrQuery();
		solrParams.set("q", "*:*");
		solrParams.set("rows", Integer.MAX_VALUE);
		QueryResponse r = null;
		try {
			r = this.server.query(solrParams);
			SolrDocumentList docs = r.getResults();
			for (SolrDocument d : docs) {
				String userId = (String) d.get("userid");
				String text = trainConnector.getTweetTextOfLastHours(userId, hours);
				tweetObjects.add(new Tweet((String) d.get("id"), userId, text, (String) d.get("timestamp"), new LinkedHashSet<String>((List<String>) d.get("hashtags"))));
			}
		} catch (SolrServerException e) {
			e.printStackTrace();
		}
		
		return tweetObjects;
	}
	
	@SuppressWarnings("unchecked")
	public Map<String, Set<String>> getUserIDs() {
		Map<String, Set<String>> tweetIDs = new LinkedHashMap<String, Set<String>>();
		
		SolrQuery solrParams = new SolrQuery();
		solrParams.set("q", "*:*");
		solrParams.set("fl", "userid,hashtags");
		solrParams.set("rows", Integer.MAX_VALUE);
		QueryResponse r = null;
		try {
			r = this.server.query(solrParams);
			SolrDocumentList docs = r.getResults();
			for (SolrDocument d : docs) {
				tweetIDs.put((String) d.get("userid"), new HashSet<String>((List<String>) d.get("hashtags")));
			}
		} catch (SolrServerException e) {
			e.printStackTrace();
		}
		
		return tweetIDs;
	}
	
	public String getMostRecentTweetOfUser(String user) {
		SolrQuery solrParams = new SolrQuery();
		solrParams.set("q", "userid:" + user);
		solrParams.set("sort", "timestamp desc");
		solrParams.set("fl", "id");
		solrParams.set("rows", 1);
		QueryResponse r = null;
		try {
			r = this.server.query(solrParams);
			SolrDocumentList docs = r.getResults();
			for (SolrDocument d : docs) {
				return (String) d.get("id");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return null;
	}
	
	public String getTweetTextOfRecentTweets(String user, int intValue) {
		String tweetText = "";
		SolrQuery solrParams = new SolrQuery();
		solrParams.set("q", "userid:" + user);
		solrParams.set("sort", "timestamp desc");
		solrParams.set("fl", "text");
		solrParams.set("rows", intValue);
		QueryResponse r = null;
		try {
			r = this.server.query(solrParams);
			SolrDocumentList docs = r.getResults();
			for (SolrDocument d : docs) {
				tweetText += ((String) d.get("text") + " ");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}		
		return tweetText;
	}
	
	public String getTweetTextOfLastHours(String user, int hours) {
		String tweetText = "";
		SolrQuery solrParams = new SolrQuery();
		solrParams.set("q", "userid:" + user);
		solrParams.set("sort", "timestamp desc");
		solrParams.set("fl", "text,timestamp");
		solrParams.set("rows", 100);
		QueryResponse r = null;
		try {
			r = this.server.query(solrParams);
			SolrDocumentList docs = r.getResults();
			Long threshold = null;
			for (SolrDocument d : docs) {
				String timestampString = (String) d.get("timestamp");
				Long timestamp = Long.parseLong(timestampString.substring(0, timestampString.indexOf(".")));
				if (threshold == null) {
					threshold = timestamp - hours * 60 * 60;
				}
				if (timestamp > threshold.longValue()) {
					tweetText += ((String) d.get("text") + " ");
				} else {
					break;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}		
		return tweetText;
	}
	
	@SuppressWarnings({ "unchecked", "deprecation" })
	public Map<String, Double> getTopHashtagsForTweetText(String tweetText, int limit) {	
		Map<String, Double> hashtagMap = new LinkedHashMap<String, Double>();
		String cleanedTweetText = tweetText;//getCleanedTweetText(tweetText);
		if (cleanedTweetText == null || cleanedTweetText.isEmpty()) {
			return hashtagMap;
		}
		
		SolrQuery solrParams = new SolrQuery();
		// query version
		//solrParams.set("q", "text:" + cleanedTweetText);
		// mlt version
		solrParams.setQueryType("/mlt");
		solrParams.set("stream.body", cleanedTweetText);
		solrParams.set("mlt.fl", "text");
		// additional parameters
		solrParams.set("fl", "hashtags,score");
		solrParams.set("mlt.mintf", "1");
		//solrParams.set("mlt.count", 50);
		solrParams.set("rows", 50);
		QueryResponse r = null;
		try {
			r = this.server.query(solrParams);
			SolrDocumentList docs = r.getResults();
			for (SolrDocument d : docs) {
				processSolrDocument(d, hashtagMap, limit);
			}
		} catch (Exception e) {
			System.out.println("Exception with tweet-text: " + cleanedTweetText);
			e.printStackTrace();
		}
		
		return MapUtil.sortByValue(hashtagMap);
	}
	
	@SuppressWarnings({ "unchecked", "deprecation" })
	public Map<String, Double> getTopHashtagsForTweetID(String tweetID, int limit) {	
		Map<String, Double> hashtagMap = new LinkedHashMap<String, Double>();
		if (tweetID == null || tweetID.isEmpty()) {
			return hashtagMap;
		}
		
		SolrQuery solrParams = new SolrQuery();
		// query version
		//solrParams.set("q", "text:" + cleanedTweetText);
		// mlt version
		solrParams.setQueryType("/mlt");
		solrParams.set("q", "id:" + tweetID);
		solrParams.set("mlt.fl", "text");
		// additional parameters
		solrParams.set("fl", "hashtags,score");
		solrParams.set("mlt.count", 50);
		solrParams.set("rows", 50);
		QueryResponse r = null;
		try {
			r = this.server.query(solrParams);
			SolrDocumentList docs = r.getResults();
			for (SolrDocument d : docs) {
				processSolrDocument(d, hashtagMap, limit);
			}
		} catch (Exception e) {
			System.out.println("Exception with tweet-id: " + tweetID);
			e.printStackTrace();
		}
		
		return hashtagMap;
	}
	
	private void processSolrDocument(SolrDocument d, Map<String, Double> hashtagMap, int limit) {
		double score = (float) d.get("score");
		List<String> hashtags = (List<String>) d.get("hashtags");
		for (String h : hashtags) {
			if (hashtagMap.size() < limit) {
				if (!hashtagMap.containsKey(h.toLowerCase())) {
					hashtagMap.put(h.toLowerCase(), score);
				}
				//Double val = hashtagMap.get(h.toLowerCase());
				//hashtagMap.put(h.toLowerCase(), val == null ? score : val.doubleValue() + score);
			} else {
				break;
			}
		}
	}
	
	private String getCleanedTweetText(String tweetText) {
		if (tweetText != null) {
			return tweetText.replaceAll("[^a-zA-Z0-9 ]+", "").trim();
		}
		return "";
	}
}
