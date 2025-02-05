use std::{cmp::min, collections::HashMap, vec};

fn main() {
    let nums = vec![-2, 0, 1, 1, 2];
    let res = ThreeSum::three_sum(nums);
    print!("{:?}", res)
}

struct ThreeSum;

impl ThreeSum {
    pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut owned_nums = nums;
        owned_nums.sort();

        let mut res = vec![];

        if owned_nums.len() < 3 {
            return res;
        }

        for n in 0..(owned_nums.len() - 2) {
            let mval = owned_nums[n];

            if mval > 0 {
                res.sort();
                res.dedup();
                return res;
            }

            let mut rp = n + 1;
            let mut lp = owned_nums.len() - 1;

            while rp != lp {
                let rval = owned_nums[rp];
                let lval = owned_nums[lp];

                let sum = rval + lval;
                if sum < -mval {
                    rp += 1;
                } else if sum > -mval {
                    lp -= 1;
                } else {
                    res.push(vec![mval, rval, lval]);
                    lp -= 1;
                }
            }
        }
        res.sort();
        res.dedup();
        res
    }
}

struct ContainerWithMostWater;

impl ContainerWithMostWater {
    pub fn max_area(height: Vec<i32>) -> i32 {
        let mut max_capacity = 0;
        for (lindex, lh) in height.iter().enumerate() {
            if lindex + 1 == height.len() {
                continue;
            }
            for (rindex, rh) in height[(lindex + 1)..].iter().enumerate() {
                let capacity = min(lh, rh) * (rindex as i32 + 1);
                if capacity > max_capacity {
                    max_capacity = capacity
                }
            }
        }

        max_capacity
    }

    pub fn max_area_v2(height: Vec<i32>) -> i32 {
        let (mut lp, mut rp) = (0, height.len() - 1);

        let mut max_capacity = 0;
        while (lp != rp) {
            let lh = height[lp];
            let rh = height[rp];

            let capacity = min(lh, rh) * ((rp - lp) as i32);
            if capacity > max_capacity {
                max_capacity = capacity;
            }

            if lh > rh {
                rp -= 1;
            } else {
                lp += 1;
            }
        }

        max_capacity
    }
}

struct MoveZones;

impl MoveZones {
    pub fn move_zeroes(nums: &mut Vec<i32>) {
        let orignal_len = nums.len();

        nums.retain(|&num| num != 0);
        let new_len = nums.len();

        let zeros = vec![0; orignal_len - new_len];

        nums.extend(zeros.iter());
    }
}

struct LongestConsecutiveSequence;

impl LongestConsecutiveSequence {
    pub fn longest_consecutive(nums: Vec<i32>) -> i32 {
        if nums.len() == 0 {
            return 0;
        }

        let mut mutable_nums = Vec::with_capacity(nums.len());
        mutable_nums.extend_from_slice(&nums);

        mutable_nums.dedup();

        mutable_nums.sort();

        let mut consecutive_counts = vec![1; nums.len()];
        let mut index = 0;

        mutable_nums.iter().reduce(|acc, cur| {
            if cur - acc == 1 {
                consecutive_counts[index] += 1;
            } else if cur == acc {
                //do nothing
            } else {
                index += 1;
            }
            cur
        });

        consecutive_counts.into_iter().max().unwrap()
    }
}

// https://leetcode.cn/problems/group-anagrams/description/
struct GroupAnagrams;

impl GroupAnagrams {
    pub fn group_anagrams(strs: Vec<String>) -> Vec<Vec<String>> {
        let mut hashmap = HashMap::<String, Vec<String>>::new();

        for word in strs {
            let mut chars: Vec<&str> = word.split("").collect();
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
