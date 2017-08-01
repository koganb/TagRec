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

import file.BookmarkReader;
import itemrecommendations.CFResourceCalculator;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import common.Bookmark;
import common.DoubleMapComparator;
import common.Features;
import common.Similarity;

// TODO: cache values
public class ResourceRecommenderEngine implements EngineInterface {

	private BookmarkReader reader = null;
	private CFResourceCalculator calculator = null;
	private CFResourceCalculator tagCalculator = null;
	private CFResourceCalculator cbCalculator = null;
	private CFResourceCalculator resCFCalculator = null;
	private final Map<Integer, Double> topResources;

	public ResourceRecommenderEngine() {
		this.topResources = new LinkedHashMap<Integer, Double>();		
		this.reader = new BookmarkReader(0, false);
	}
	
	public void loadFile(String path, String filename) throws Exception {
		BookmarkReader reader = EngineUtils.getSortedBookmarkReader(path, filename);

		CFResourceCalculator calculator = new CFResourceCalculator(reader, reader.getBookmarks().size(), false, true, false, 5, Similarity.COSINE, Features.ENTITIES);
		CFResourceCalculator tagCalculator = new CFResourceCalculator(reader, reader.getBookmarks().size(), false, true, false, 5, Similarity.COSINE, Features.TAGS);
		CFResourceCalculator cbCalculator = new CFResourceCalculator(reader, reader.getBookmarks().size(), false, false, true, 5, Similarity.COSINE, Features.TAGS);
		CFResourceCalculator resCFCalculator = new CFResourceCalculator(reader, reader.getBookmarks().size(), false, false, true, 5, Similarity.COSINE, Features.ENTITIES);
		
		Map<Integer, Double> topResources = EngineUtils.calcTopEntities(reader, EntityType.RESOURCE);
		resetStructure(reader, calculator, tagCalculator, cbCalculator, resCFCalculator, topResources);
	}

	public synchronized Map<String, Double> getEntitiesWithLikelihood(String user, String resource, List<String> topics, Integer count, Boolean filterOwnEntities, Algorithm algorithm, EntityType type) {
		if (count == null || count.doubleValue() < 1) {
			count = 10;
		}
		if (filterOwnEntities == null) {
			filterOwnEntities = true;
		}
		
		Map<Integer, Double> resourceIDs = new LinkedHashMap<>();
		Map<String, Double> resourceMap = new LinkedHashMap<>();
		if (this.reader == null || this.calculator == null) {
			System.out.println("No data has been loaded");
			return resourceMap;
		}
		int userID = -1;
		if (user != null) {
			userID = this.reader.getUsers().indexOf(user);
		}
		int resID = -1;
		if (resource != null) {
			resID = this.reader.getResources().indexOf(resource);
		}
		// used to filter own resources if necessary
		List<Integer> userResources = null;
		if (userID != -1 && filterOwnEntities.booleanValue()) {
			userResources = Bookmark.getResourcesFromUser(this.reader.getBookmarks(), userID);
		}

		if (algorithm == null || algorithm != Algorithm.RESOURCEMP) {
			if (userID != -1) {
					if (algorithm == Algorithm.RESOURCETAGCF) {
						resourceIDs = this.tagCalculator.getRankedResourcesList(userID, -1, false, false, false, filterOwnEntities.booleanValue(), false); // not sorted!
					} else if (algorithm == Algorithm.RESOURCETAGCB)  {
						resourceIDs = this.cbCalculator.getRankedResourcesList(userID, -1, false, false, false, filterOwnEntities.booleanValue(), false); // not sorted!
					} else {
						resourceIDs = this.calculator.getRankedResourcesList(userID, -1, false, false, false, filterOwnEntities.booleanValue(), false); // not sorted!
					}
			} else if (resID != -1) {
				if (algorithm == Algorithm.RESOURCETAGCF) {
					resourceIDs = this.cbCalculator.getRankedResourcesList(-1, resID, false, false, false, filterOwnEntities.booleanValue(), false); // not sorted!
				} else {
					resourceIDs = this.resCFCalculator.getRankedResourcesList(-1, resID, false, false, false, filterOwnEntities.booleanValue(), false); // not sorted!
				}
			}
		}
		// then call MP if necessary
		if (resourceIDs.size() < count) {
			for (Map.Entry<Integer, Double> t : this.topResources.entrySet()) {
				if (resourceIDs.size() < count) {
					// add MP resources if they are not already in the recommeded list or already known by this user
					if (!resourceIDs.containsKey(t.getKey()) && (userResources == null || !userResources.contains(t.getKey()))) {
						resourceIDs.put(t.getKey(), t.getValue());
					}
				} else {
					break;
				}
			}
		}

		// sort
		Map<Integer, Double> sortedResultMap = new TreeMap<Integer, Double>(new DoubleMapComparator(resourceIDs));
		sortedResultMap.putAll(resourceIDs);
		
		// last map IDs back to strings
		for (Map.Entry<Integer, Double> tEntry : sortedResultMap.entrySet()) {
			if (resourceMap.size() < count) {
				resourceMap.put(this.reader.getResources().get(tEntry.getKey()), tEntry.getValue());
			} else {
				break;
			}
		}
		
		return resourceMap;
	}

	public synchronized void resetStructure(BookmarkReader reader, CFResourceCalculator calculator, 
			CFResourceCalculator tagCalculator, CFResourceCalculator cbCalculator, CFResourceCalculator resCFCalculator, Map<Integer, Double> topResources) {
		this.reader = reader;
		this.calculator = calculator;
		this.tagCalculator = tagCalculator;
		this.cbCalculator = cbCalculator;
		this.resCFCalculator = resCFCalculator;
		
		this.topResources.clear();
		this.topResources.putAll(topResources);
	}
}
