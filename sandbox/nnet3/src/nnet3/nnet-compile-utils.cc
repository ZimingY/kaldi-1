// Copyright      2015  Johns Hopkins University (author: Daniel Povey)
//                2015                           (author: Vijayaditya Peddinti)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <iterator>
#include <sstream>
#include "util/common-utils.h"
#include "nnet3/nnet-compile-utils.h"

namespace kaldi {
namespace nnet3 {

// this comparator will be used to sort/find pairs using first_element
// we declare it as a struct as it will also be used by std::lower_bound
// method which will supply elements of different types to the function
struct FirstElementComparator {
  int first_element(int32 t) const {
    return t;
  }

  int first_element(std::pair<int32, int32> t) const  {
    return t.first;
  }

  template< typename T1, typename T2>
  bool operator()( T1 const & t1, T2 const & t2) const  {
    return first_element(t1) < first_element(t2);
  }
};

// this comparator will be used to sort pairs initially by second element in
// ascending order and then by first element in descending order
bool SecondElementComparator(const std::pair<int32, int32>& first_pair,
                               const std::pair<int32, int32>& second_pair) {
    if (first_pair.second == second_pair.second)
      return first_pair.first < second_pair.first;
    return first_pair.second > second_pair.second;
}

// function to compute a histogram of the first_elements in the pairs of vectors
// of lists of pairs. It also generates a copy of submat_lists where the lists
// are sorted (this will be used in the caller function to do sort and find
// operations effeciently). While computing the histogram multiple occurrences
// of the same first_element are treated as separate symbols. It also outputs
// the max_submat_list_size which is the maximum size of the lists in the vector
void ComputeHistogram(
    const std::vector< std::vector<std::pair<int32, int32> > > submat_lists,
    std::vector<std::vector<std::pair<int32, int32> > > *sorted_submat_lists,
    unordered_map<int32, std::vector<int32> > *submat_histogram,
    int32 *max_submat_list_size) {
  // computing the submat_histogram
  for (int32 i = 0; i < submat_lists.size(); i++) {
    if (submat_lists[i].size() > *max_submat_list_size)
      *max_submat_list_size = submat_lists[i].size();
    sorted_submat_lists->at(i) = submat_lists[i];
    std::sort(sorted_submat_lists->at(i).begin(),
              sorted_submat_lists->at(i).end(),
              FirstElementComparator());
    // counting the occurrences of each element in the current submat_list
    // each new occurrence of the same element, in this list, is counted
    // seperately
    int j = 0;
    unordered_map<int32, std::vector<int32> >::iterator histogram_iterator
        = submat_histogram->end();
    int32 repetition_count = 0;
    while (j < sorted_submat_lists->at(i).size()) {
      if ((histogram_iterator == submat_histogram->end()) ||
          (histogram_iterator->first != submat_lists[i][j].first)) {
        histogram_iterator = submat_histogram->find(submat_lists[i][j].first);
        repetition_count = 0;
        // if a histogram was not found for this submat_index, add one
        if (histogram_iterator == submat_histogram->end()) {
          (*submat_histogram)[submat_lists[i][j].first];
          histogram_iterator = submat_histogram->find(submat_lists[i][j].first);
        }
      }

      if (repetition_count >= (histogram_iterator->second).size()) {
        // this is the first time the symbol repeated this many times
        (histogram_iterator->second).push_back(1);
      } else {
        (histogram_iterator->second)[repetition_count]++;
      }
      repetition_count++;
    }
  }
}

// function which does the actual splitting of submat_lists. But it operates on
// sorted submat_lists and uses submat_histogram_vector. (See
// nnet-compile-utils.h for description of the actual SplitLocations)
void SplitLocations(
    // maximum size of the list in the sorted_submat_lists
    int32 max_submat_list_size,
    // second level vectors are expected to be sorted in ascending order by
    // first element
    std::vector<std::vector<std::pair<int32, int32> > > *sorted_submat_lists,
    std::vector<std::pair<int32, int32> > *submat_histogram_vector,
    // this is an empty vector where output is store
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists
    )  {
  
  // sort the submat_histogram_vector based on second element of pair
  // in ascending order then first element of pair in descending order
  std::sort(submat_histogram_vector->begin(),
            submat_histogram_vector->end(), SecondElementComparator);
  int32 prev_max_remaining_submat_list_size = max_submat_list_size;
  while (submat_histogram_vector->size() > 0)  {
    std::pair<int32, int32> submat_index_and_count =
        submat_histogram_vector->back();
    submat_histogram_vector->pop_back();
    std::vector<std::vector<std::pair<int32, int32> >::iterator>
        output_iterator_list;
    output_iterator_list.reserve(sorted_submat_lists->size());

    // go through the submat_lists and find out the
    // max_remaining_submat_list_size if one occurrence of
    // the current submat_index was removed from the lists
    int32 max_remaining_submat_list_size = 0;
    for (int32 i = 0; i < sorted_submat_lists->size(); i++)  {
      std::vector< std::pair<int32, int32> > & submat_list =
          sorted_submat_lists->at(i);
      output_iterator_list.push_back(
          std::lower_bound(submat_list.begin(), submat_list.end(),
                           submat_index_and_count.first,
                           FirstElementComparator()));
      int32 remaining_submat_list_size = 0;
      if  (output_iterator_list.back() != submat_list.end())  {
        // since the submat_index is present in this submat_list
        // if submat_index was deleted from the list
        // the remaining submat_list's size is reduced by 1
        remaining_submat_list_size--;
      }
      max_remaining_submat_list_size = remaining_submat_list_size
          > max_remaining_submat_list_size ? remaining_submat_list_size :
          max_remaining_submat_list_size;
    }
    
    if (max_remaining_submat_list_size
        <= prev_max_remaining_submat_list_size)  {
      // since we will have a smaller max_remaining_submat_list_size by
      // splitting this submat_index, we will split it into a seperate list;
      // this submat_index will be accessed using a kAddRows call;
      // for this we go through the list of vector iterators and pop the
      // elements wherever available
      // Note : To decide splits we could have solved a combinatorial 
      // optimization problem where we find the best set of
      // kAddRows + kAddRowsMulti calls;
      // but given that both these calls have similar costs,
      // and that the average number of elements in a list is around 4,
      // it does not make sense to
      // choose a kAddRows call unless it is able to immediately reduce a
      // kAddRowsMulti call. So we simplify the process and stay away from any
      // complex optimization algorithms.
      // Implement dynamic programming based solution,
      // if the length of the lists increases for newer NN architectures

      std::vector<std::pair<int32, int32> > list_of_pairs;
      list_of_pairs.reserve(sorted_submat_lists->size());
      for (int32 i = 0; i < output_iterator_list.size(); i++) {
        if (output_iterator_list[i] != sorted_submat_lists->at(i).end()) {
          // there was an element with the submat_index in the current list
          list_of_pairs.push_back(*output_iterator_list[i]);
          sorted_submat_lists->at(i).erase(output_iterator_list[i]);
        } else  {
          // insert a dummy element. Callers of this function expect the dummy
          // element to be (-1, -1)
          list_of_pairs.push_back(std::make_pair(-1, -1));
        }
      }
      split_lists->push_back(list_of_pairs);
    }
    prev_max_remaining_submat_list_size = max_remaining_submat_list_size;
  }

  // combine the remaining pairs into lists where
  // paris with multiple first elements are allowed 
  for (int32 i = 0; i < prev_max_remaining_submat_list_size; i++) {
    std::vector<std::pair<int32, int32> > list_of_pairs;
    list_of_pairs.reserve(sorted_submat_lists->size());
    for (int32 j = 0; j < sorted_submat_lists->size(); j++)  {
      if (sorted_submat_lists[i].size() == 0)  {
        list_of_pairs.push_back(std::make_pair(-1, -1));
        continue;
      }
      // Note : we don't yet know if there is any advantage of having multiple
      // calls to the same submat in kAddRowsMulti. If this is actually helpful
      // then use the sorted_histogram_vector to first copy submat_indexes which
      // did not make it to kAddRows calls
      list_of_pairs.push_back(sorted_submat_lists->at(i).back());
      sorted_submat_lists->at(i).pop_back();
    }
    split_lists->push_back(list_of_pairs);
  }
}

void SplitLocations(
    const std::vector<std::vector<std::pair<int32, int32> > > &submat_lists,
    std::vector<std::vector<std::pair<int32, int32> > > *split_lists) {

  // computing a histogram of the submat_indexes in the submat_lists
  // each occurence in a given list is considered unique so we maintain
  // a vector to count each occurence seperately
  unordered_map<int32, std::vector<int32> > submat_histogram;
  // declaring a variable to store the max_submat_list_size
  // this is equal to the number of split_lists necessary to store the pairs
  // if just split the submat_lists without optimization
  int32 max_submat_list_size = 0;

  // initializing a vector of list of pairs which is mutable
  // and where the sorted submat_lists are sorted, for faster search
  std::vector<std::vector< std::pair<int32, int32> > >
      sorted_submat_lists;
  sorted_submat_lists.reserve(submat_lists.size());
  std::vector<std::pair<int32, int32> > submat_histogram_vector;

  ComputeHistogram(submat_lists, &sorted_submat_lists,
                   &submat_histogram, &max_submat_list_size);

  // copy the key, occurence_counts from submat_histogram to a vector
  unordered_map<int32, std::vector<int32> >::iterator hist_iter;
  for (hist_iter = submat_histogram.begin();
       hist_iter != submat_histogram.end();
       ++hist_iter) {
    for (int32 i = 0; i < (hist_iter->second).size(); i++)  {
      submat_histogram_vector.push_back(
          std::make_pair(hist_iter->first, (hist_iter->second)[i]));
    }
  }

  SplitLocations(max_submat_list_size, &sorted_submat_lists,
                 &submat_histogram_vector, split_lists);
}

bool ConvertToIndexes(
    const std::vector<std::pair<int32, int32> > &location_vector,
    int32 *first_value,
    std::vector<int32> *second_values)  {
  *first_value = -1;
  second_values->reserve(location_vector.size());
  std::vector<std::pair<int32, int32> >::const_iterator iter;
  for (iter = location_vector.begin(); iter < location_vector.end(); iter++)  {
    if (iter->first != -1) {
      if (*first_value == -1)
        *first_value = iter->first;
      if (iter->first != *first_value)
        return false;
      second_values->push_back(iter->second);
    } else  {
      second_values->push_back(-1);
    }
  }
  return true;
}

}  // namespace nnet3
}  // namespace kaldi
