use std::{collections::HashMap, vec};

fn main() {
    let res = GroupAnagrams::group_anagrams(vec![
        "eat".into(),
        "tea".into(),
        "tan".into(),
        "ate".into(),
        "nat".into(),
        "bat".into(),
    ]);
    println!("{:?}", res)
}

// https://leetcode.cn/problems/group-anagrams/description/
struct GroupAnagrams;

impl GroupAnagrams {
    pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
        let mut hashmap = HashMap::<String, Vec<String>>::new();

        for word in strs {
            let mut chars: Vec<String> = word.chars().collect();
            chars.sort();

            let sorted_word = chars.join("");

            match hashmap.get_mut(&sorted_word) {
                Some(set) => {
                    set.push(word);
                }
                None => {
                    hashmap.insert(sorted_word, vec![word]);
                }
            }
        }

        hashmap.into_values().collect::<Vec<Vec<String>>>()
    }

    pub fn group_anagrams_v2(strs: Vec<String>) -> Vec<Vec<String>> {
        let mut hashmap: HashMap<Vec<i32>, Vec<String>> = HashMap::new();

        for word in strs {
            let mut count = vec![0; 26]; // Assuming lowercase a-z
            for c in word.chars() {
                count[c as usize - 'a' as usize] += 1;
            }

            hashmap.entry(count).or_insert_with(Vec::new).push(word);
        }

        hashmap.into_values().collect()
    }
}

impl GroupAnagrams {}

// https://leetcode.cn/problems/two-sum/description/
struct TwoSum;

impl TwoSum {
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        let nums_with_index: Vec<(usize, i32)> = nums.into_iter().enumerate().collect();

        // For each l_num, check the corresponding r_num existence
        for (l_index, l_num) in &nums_with_index {
            let expect = target - *l_num;
            // Each value can be used only once
            match nums_with_index.iter().find(|tuple| tuple.1 == expect) {
                Some((r_index, _)) => {
                    if *r_index != *l_index {
                        return vec![(*l_index) as i32, (*r_index) as i32];
                    }
                }
                None => {}
            }
        }

        vec![]
    }

    pub fn two_sum_v2(nums: Vec<i32>, target: i32) -> Vec<i32> {
        let mut hashmap: HashMap<i32, usize> = HashMap::new();

        let nums_with_index: Vec<(usize, i32)> = nums.into_iter().enumerate().collect();

        for (index, num) in &nums_with_index {
            hashmap.insert(*num, *index);
        }
        println!("{:?}", hashmap);
        for (l_index, l_num) in &nums_with_index {
            let r_num = target - *l_num;
            match hashmap.get(&r_num) {
                Some(r_index) => {
                    if *r_index != *l_index {
                        return vec![*l_index as i32, *r_index as i32];
                    }
                }
                None => {}
            }
        }

        vec![]
    }
}
