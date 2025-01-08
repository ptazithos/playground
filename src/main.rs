use std::collections::HashMap;

fn main() {
    let res = TwoSum::two_sum_v2(vec![3, 2, 3], 6);
    println!("{:?}", res)
}

//https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked
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
