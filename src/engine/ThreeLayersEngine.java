/*
 TagRecommender:
 A framework to implement and evaluate algorithms for the recommendation
 of tags.
 Copyright (C) 2013 Dominik Kowald
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.
 
 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package engine;

import processing.ThreeLTCalculator;
import file.BookmarkReader;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import common.CalculationType;
import common.DoubleMapComparator;

// TODO: make it work in online setting! (caching + LDA topic calculation)
public class ThreeLayersEngine implements EngineInterface {

	private BookmarkReader reader = null;
	private ThreeLTCalculator calculator = null;
	private final Map<Integer, Double> topTags;

	public ThreeLayersEngine() {
		this.topTags = new LinkedHashMap<>();		
		this.reader = null;
	}
	
	public void loadFile(String path, String filename) throws Exception {
		BookmarkReader reader = EngineUtils.getSortedBookmarkReader(path, filename);

		ThreeLTCalculator calculator = new ThreeLTCalculator(reader, reader.getBookmarks().size(), 5, 5, true, true, false, CalculationType.NONE);
		Map<Integer, Double> topTags = EngineUtils.calcTopEntities(reader, EntityType.TAG);
		resetStructure(reader, calculator, topTags);
	}

	public synchronized Map<String, Double> getEntitiesWithLikelihood(String user, String resource, List<String> topics, Integer count, Boolean filterOwnEntities, Algorithm algorithm, EntityType type) {
		if (count == null || count.doubleValue() < 1) {
			count = 10;
		}
		if (filterOwnEntities == null) {
			filterOwnEntities = true;
		}
		List<Integer> filterTags = new ArrayList<Integer>();
		
		Map<Integer, Double> tagIDs = new LinkedHashMap<>();
		Map<String, Double> tagMap = new LinkedHashMap<>();
		if (this.reader == null || this.calculator == null) {
			return tagMap;
		}
		if (algorithm == null || algorithm != Algorithm.MP) {			
			int userID = -1;
			if (user != null) {
				userID = this.reader.getUsers().indexOf(user);
			}
			filterTags = EngineUtils.getFilterTags(filterOwnEntities, this.reader, user, resource/*, this.calculator.getUserMaps().get(userID)*/);
			int resID = -1;
			if (resource != null) {
				resID = this.reader.getResources().indexOf(resource);
			}
			List<Integer> topicIDs = new ArrayList<>();
			if (topics != null) {
				for (String t : topics) {
					int tID = this.reader.getCategories().indexOf(t);
					if (tID != -1) {
						topicIDs.add(tID);
					}
				}
			}
			if (algorithm == null || algorithm == Algorithm.THREELTMPr) {
				tagIDs = this.calculator.getRankedTagList(userID, resID, topicIDs, System.currentTimeMillis() / 1000.0, count, this.reader.hasTimestamp(), false, false); // not sorted
			} else if (algorithm == Algorithm.THREELT) {
				tagIDs = this.calculator.getRankedTagList(userID, -1, topicIDs, System.currentTimeMillis() / 1000.0, count, this.reader.hasTimestamp(), false, false); // not sorted
			} else if (algorithm == Algorithm.THREEL) {
				tagIDs = this.calculator.getRankedTagList(userID, -1, topicIDs, System.currentTimeMillis() / 1000.0, count, false, false, false); // not sorted
			}
		}
		
		// TODO: finish filtering
		
		// fill up with MP tags
		if (tagIDs.size() < count) {
			for (Map.Entry<Integer, Double> t : this.topTags.entrySet()) {
				if (tagIDs.size() < count) {
					if (!tagIDs.containsKey(t.getKey())) {
						tagIDs.put(t.getKey(), t.getValue());
					}
				} else {
					break;
				}
			}
		}
		
		// sort
		Map<Integer, Double> sortedResultMap = new TreeMap<Integer, Double>(new DoubleMapComparator(tagIDs));
		sortedResultMap.putAll(tagIDs);
		// map tag-IDs back to strings
		for (Map.Entry<Integer, Double> tEntry : sortedResultMap.entrySet()) {
			if (tagMap.size() < count) {
				tagMap.put(this.reader.getTags().get(tEntry.getKey()), tEntry.getValue());
			}
		}
		return tagMap;
	}

	public synchronized void resetStructure(BookmarkReader reader, ThreeLTCalculator calculator, Map<Integer, Double> topTags) {
		this.reader = reader;
		this.calculator = calculator;
		
		this.topTags.clear();
		this.topTags.putAll(topTags);
	}
}
