#![allow(unused)]
use std::{
    cell::RefCell,
    cmp::{max, min},
    collections::{HashMap, VecDeque},
    mem::swap,
    rc::Rc,
    vec,
};

fn main() {
    let mut res = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
    Solution::rotate_image(&mut res);
    print!("{:?}", res);
}

// Definition for singly-linked list.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}
impl Solution {
    pub fn is_palindrome(head: Option<Box<ListNode>>) -> bool {
        let mut deque = VecDeque::new();
        let mut p = head.clone();
        while let Some(node) = p {
            deque.push_back(node.val);

            p = node.next;
        }

        let mut p = head.clone();
        while let Some(node) = p {
            let back = deque.pop_back().unwrap();
            if back != node.val {
                return false;
            }
            p = node.next;
        }

        return true;
    }
}

impl Solution {
    pub fn rotate_image(matrix: &mut Vec<Vec<i32>>) {
        let n = matrix.len();
        for i in 0..n {
            for j in 0..i {
                let rv = matrix[i][j];
                let lv = matrix[j][i];

                matrix[j][i] = rv;
                matrix[i][j] = lv;
                println!("i:{:?}, j:{:?},{:?}", i, j, matrix);
            }
        }

        for i in 0..n {
            matrix[i].reverse();
        }
    }
}

impl Solution {
    pub fn spiral_order(matrix: Vec<Vec<i32>>) -> Vec<i32> {
        let mut matrix = matrix;

        let height = matrix.len();
        let width = matrix[0].len();

        let count = (std::cmp::min(width, height) + 1) / 2;

        let mut res: Vec<i32> = vec![];
        for i in 0..count {
            let top_row: Vec<i32> = matrix.remove(0);
            println!("top row: {:?}", top_row);
            if matrix.len() == 0 {
                res.extend(top_row);
                break;
            }
            let mut bottom_row: Vec<i32> = matrix.remove(matrix.len() - 1);
            println!("bottom row: {:?}", bottom_row);

            let mut left_col: Vec<i32> = vec![];
            let mut right_col: Vec<i32> = vec![];
            for j in 0..matrix.len() {
                let mut temp_row = &mut matrix[j];
                let tail = temp_row.remove(temp_row.len() - 1);

                right_col.push(tail);
                if temp_row.len() == 0 {
                    continue;
                }
                let head = temp_row.remove(0);

                left_col.push(head);
            }
            println!("left col: {:?}", left_col);
            println!("right col: {:?}", right_col);

            res.extend(top_row);
            res.extend(right_col);
            bottom_row.reverse();
            res.extend(bottom_row);
            left_col.reverse();
            res.extend(left_col);
        }

        res
    }
}

impl Solution {
    pub fn set_zeroes(matrix: &mut Vec<Vec<i32>>) {
        let mut marker_hashmap: HashMap<usize, bool> = HashMap::new();
        for i in 0..matrix.len() {
            for j in 0..matrix[i].len() {
                if matrix[i][j] == 0 {
                    marker_hashmap.entry(i).or_insert(true);
                    marker_hashmap.entry(1000 + j).or_insert(true);
                }
            }
        }

        for i in 0..matrix.len() {
            for j in 0..matrix[i].len() {
                if let Some(marker) = marker_hashmap.get(&i) {
                    matrix[i][j] = 0;
                }

                if let Some(marker) = marker_hashmap.get(&(1000 + j)) {
                    matrix[i][j] = 0;
                }
            }
        }
    }
}

struct Solution {}
type Res = Rc<RefCell<Vec<String>>>;
type Map = HashMap<String, Vec<String>>;

impl Solution {
    pub fn recursive_concat(res: Res, map: &Map, digits: &String, depth: usize, comb: String) {
        if depth == digits.len() {
            if comb.len() > 0 {
                res.borrow_mut().push(comb.clone());
            }
            return;
        }

        let digit = digits.chars().nth(depth).unwrap().to_string();
        let chars = map.get(&digit).unwrap();

        chars.iter().for_each(|char| {
            let mut comb_cloned = comb.clone();
            comb_cloned.push_str(char);

            let res_cloned = res.clone();
            Self::recursive_concat(res_cloned, map, digits, depth + 1, comb_cloned);
        });
    }

    pub fn letter_combinations(digits: String) -> Vec<String> {
        let mut map: Map = HashMap::new();
        map.insert("2".into(), vec!["a".into(), "b".into(), "c".into()]);
        map.insert("3".into(), vec!["d".into(), "e".into(), "f".into()]);
        map.insert("4".into(), vec!["g".into(), "h".into(), "i".into()]);
        map.insert("5".into(), vec!["j".into(), "k".into(), "l".into()]);
        map.insert("6".into(), vec!["m".into(), "n".into(), "o".into()]);
        map.insert(
            "7".into(),
            vec!["p".into(), "q".into(), "r".into(), "s".into()],
        );
        map.insert("8".into(), vec!["t".into(), "u".into(), "v".into()]);
        map.insert(
            "9".into(),
            vec!["w".into(), "x".into(), "y".into(), "z".into()],
        );

        let res: Res = Rc::new(RefCell::new(vec![]));

        Self::recursive_concat(res.clone(), &map, &digits, 0, String::from(""));

        res.take()
    }
}

impl Solution {
    pub fn first_missing_positive(nums: Vec<i32>) -> i32 {
        let mut nums = nums;
        nums.sort();
        for i in 0..nums.len() {
            let num = nums[i];
            if num > 0 && num < nums.len() as i32 + 1 {
                if num != i as i32 + 1 {
                    nums.swap(i, num as usize - 1);
                }
            }
        }

        println!("{:?}", nums);
        for i in 0..nums.len() {
            if nums[i] as usize != i + 1 {
                return (i + 1) as i32;
            }
        }

        (nums.len() + 1) as i32
    }
}

impl Solution {
    pub fn rotate(nums: &mut Vec<i32>, k: i32) {
        let mut nums_deque: VecDeque<i32> = nums.drain(..).collect();

        for _ in 0..k {
            let back = nums_deque.pop_back().unwrap();
            nums_deque.push_front(back);
        }

        let mut nums_vec: Vec<i32> = nums_deque.into_iter().collect();
        swap(nums, &mut nums_vec);
    }
}

impl Solution {
    pub fn max_sliding_window(nums: Vec<i32>, k: i32) -> Vec<i32> {
        //handle invalid inputs
        if nums.len() == 0 || k == 0 {
            return vec![];
        }

        let mut mono_desc_deque = VecDeque::<usize>::new();
        let mut res: Vec<i32> = vec![];

        for (index, num) in nums.iter().enumerate() {
            //Remove the elements less than it from deque
            while mono_desc_deque.len() > 0
                && *num > nums[*mono_desc_deque.back().unwrap() as usize]
            {
                mono_desc_deque.pop_back();
            }

            //Remove the heading element if out of the window range
            if mono_desc_deque.len() > 0
                && (*mono_desc_deque.front().unwrap() as i32) <= index as i32 - k
            {
                mono_desc_deque.pop_front();
            }

            //Push the number at the end of the deque
            mono_desc_deque.push_back(index);

            if (index as i32) >= k - 1 {
                res.push(nums[*mono_desc_deque.front().unwrap()]);
            }
        }

        res
    }
}

impl Solution {
    pub fn subarray_sum(nums: Vec<i32>, k: i32) -> i32 {
        let mut presum_map: HashMap<i32, i32> = HashMap::new();
        let mut res = 0;

        nums.into_iter()
            .enumerate()
            .fold(0, |presum, (index, num)| {
                let presum = num + presum;

                if presum == k {
                    res += 1;
                }

                if presum_map.contains_key(&(presum - k)) {
                    let count = presum_map.get(&(presum - k)).unwrap();
                    res += *count;
                }

                presum_map
                    .entry(presum)
                    .and_modify(|num| *num += 1)
                    .or_insert(1);

                presum
            });

        res
    }
}

impl Solution {
    pub fn find_anagrams(s: String, p: String) -> Vec<i32> {
        if p.len() > s.len() {
            return vec![];
        }

        let ascii_base = 'a' as usize;
        let mut p_chars = [0; 26];
        for c in p.chars() {
            let ascii_code = c.to_ascii_lowercase() as usize;
            let pos = ascii_code - ascii_base;

            p_chars[pos] += 1;
        }

        let mut res = vec![];
        for i in 0..(s.len() - p.len() + 1) {
            let mut s_chars = [0; 26];
            for c in s[i..i + p.len()].chars() {
                let ascii_code = c.to_ascii_lowercase() as usize;
                let pos = ascii_code - ascii_base;
                s_chars[pos] += 1;
            }
            if p_chars == s_chars {
                res.push(i as i32);
            }
        }

        res
    }
}

impl Solution {
    pub fn length_of_longest_substring(s: String) -> i32 {
        let chars: Vec<char> = s.chars().collect();

        if chars.len() == 0 {
            return 0;
        }

        let mut lp = 0;
        let mut rp = 1;

        let mut candidate = String::from(&s[lp..rp]);
        let mut res = candidate.len();

        while rp != s.len() {
            let new_char = chars[rp];
            if candidate.contains(new_char) {
                lp += 1;
                rp = max(lp + 1, rp);
            } else {
                rp += 1;
            }
            candidate = String::from(&s[lp..rp]);
            if candidate.len() > res {
                res = candidate.len();
            }
        }

        res as i32
    }
}

impl Solution {
    pub fn trap(height: Vec<i32>) -> i32 {
        let mut max_left = 0;
        let mut max_lefts = height
            .iter()
            .map(|h| {
                if *h > max_left {
                    max_left = *h;
                }
                return max_left;
            })
            .collect::<Vec<i32>>();

        max_lefts.insert(0, 0);
        max_lefts.pop();

        let mut max_right = 0;
        let mut max_rights = height
            .iter()
            .rev()
            .map(|h| {
                if *h > max_right {
                    max_right = *h
                }

                max_right
            })
            .collect::<Vec<i32>>();

        max_rights.insert(0, 0);
        max_rights.pop();
        max_rights.reverse();

        let mut capacity = 0;

        for (index, h) in height.iter().enumerate() {
            capacity += max(min(max_lefts[index], max_rights[index]) - *h, 0);
        }

        capacity
    }
}

impl Solution {
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

impl Solution {
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

impl Solution {
    pub fn move_zeroes(nums: &mut Vec<i32>) {
        let original_len = nums.len();

        nums.retain(|&num| num != 0);
        let new_len = nums.len();

        let zeros = vec![0; original_len - new_len];

        nums.extend(zeros.iter());
    }
}

impl Solution {
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

impl Solution {
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

// https://leetcode.cn/problems/two-sum/description/

impl Solution {
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
