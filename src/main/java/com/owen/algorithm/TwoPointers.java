package com.owen.algorithm;

import java.util.*;

public class TwoPointers {

    //[3].无重复子串的最长子串（双指针）
    public static int lengthOfLongestSubstring(String s) {
        //滑动窗口
        int[] position = new int[128];
        int start = 0, end = 0;
        int result = 0;
        char[] array = s.toCharArray();
        while (end < array.length) {
            char character = array[end];
            int lastMaxIndex = position[character];
            //滑动窗口缩小
            start = Math.max(start, lastMaxIndex);
            //更新最长子串的长度
            result = Math.max(result, end - start + 1);
            //更新字符的最大位置
            position[character] = end + 1;
            //滑动窗口扩大
            end++;
        }
        return result;
    }

    //[3].无重复子串的最长子串（滑动窗口）
    public static int lengthOfLongestSubstringV2(String s) {
        Map<Character, Integer> window = new HashMap<>();

        int right = 0, left = 0, res = 0;
        while (right < s.length()) {
            char c = s.charAt(right);
            window.put(c, window.getOrDefault(c, 0) + 1);
            right++;
            //缩窗口条件: 有重复字符
            while (window.get(c) > 1) {
                char d = s.charAt(left);
                left++;
                window.put(d, window.get(d) - 1);
            }
            res = Math.max(res, right - left);
        }
        return res;
    }

    //[76].最小覆盖子串
    public static String minWindow(String s, String t) {
        Map<Character, Integer> need = new HashMap<>();
        for (char ch : t.toCharArray()) {
            need.put(ch, need.getOrDefault(ch, 0) + 1);
        }

        Map<Character, Integer> window = new HashMap<>();

        int right = 0, left = 0, valid = 0, start = 0, len = Integer.MAX_VALUE;
        while (right < s.length()) {
            char ch = s.charAt(right);
            right++;

            if (need.containsKey(ch)) {
                window.put(ch, window.getOrDefault(ch, 0) + 1);
                if (window.get(ch).equals(need.get(ch))) {
                    valid++;
                }
            }

            //缩窗口
            while (valid == need.size()) {
                if (right - left < len) {
                    len = right - left;
                    start = left;
                }

                char d = s.charAt(left);
                left++;

                if (need.containsKey(d)) {
                    if (window.get(d).equals(need.get(d))) {
                        valid--;
                    }
                    window.put(d, window.get(d) - 1);
                }
            }
        }
        return len == Integer.MAX_VALUE ? "" : s.substring(start, start + len);
    }

    //[209].长度最小的子数组
    public static int minSubArrayLen(int s, int[] nums) {
        int res = Integer.MAX_VALUE;
        int sum = 0;
        for (int left = 0, right = 0; right < nums.length; right++) {
            sum += nums[right];

            //一旦窗口满足，则移动起始节点，直到找到窗口不满足条件为止
            while (sum >= s) {
                res = Math.min(res, right - left + 1);
                sum -= nums[left++];
            }

        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }

    //[239].滑动窗口最大值
    public static int[] maxSlidingWindow(int[] nums, int k) {
        List<Integer> res = new ArrayList<>();
        LinkedList<Integer> monotonic = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            if (i < k - 1) {
                //先填满k-1个数字
                while (!monotonic.isEmpty() && monotonic.peekLast() < nums[i]) {
                    monotonic.pollLast();
                }
                monotonic.add(nums[i]);
            } else {
                //添加新数字
                while (!monotonic.isEmpty() && monotonic.peekLast() < nums[i]) {
                    monotonic.pollLast();
                }
                monotonic.add(nums[i]);

                //取最大值
                res.add(monotonic.getFirst());

                //移出旧数字
                if (monotonic.getFirst() == nums[i - k + 1]) {
                    monotonic.pollFirst();
                }
            }
        }
        int[] arr = new int[res.size()];
        int i = 0;
        for (Integer num : res) {
            arr[i++] = num;
        }
        return arr;
    }

    //[283].移动零
    public static void moveZeroes(int[] nums) {
        int slow = 0, fast = 0;
        while (fast < nums.length) {
            if (nums[fast] != 0) {
                nums[slow] = nums[fast];
                slow++;
            }
            fast++;
        }
        for (int i = slow; i < nums.length; i++) {
            nums[i] = 0;
        }
    }

    //[287].寻找重复数(双指针，判断链表有环)
    public static int findDuplicateV2(int[] nums) {
        int slow = 0, fast = 0;
        do {
            slow = nums[slow];
            fast = nums[nums[fast]];
        } while (slow != fast);
        slow = 0;

        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }

    //[395].至少有k个重复字符的最长子串
    public static int longestSubstring(String s, int k) {
        if (s == null || s.length() == 0) return 0;

        int[] count = new int[26];
        for (char ch : s.toCharArray()) {
            count[ch - 'a']++;
        }

        //判断这里面的每个字符的个数是否都是 >= k。是就返回。 否就分治。
        boolean full = true;
        for (char ch : s.toCharArray()) {
            if (count[ch - 'a'] > 0 && count[ch - 'a'] < k) {
                full = false;
                break;
            }
        }
        if (full) {
            return s.length();
        }

        int right = 0, left = 0, maxLen = 0;
        while (right < s.length()) {
            //当发现了字符频率小于k的就缩窗口，否则扩大
            if (count[s.charAt(right) - 'a'] < k) {
                maxLen = Math.max(maxLen, longestSubstring(s.substring(left, right), k));
                left = right + 1;
            }
            right++;
        }

        //最后一块
        maxLen = Math.max(maxLen, longestSubstring(s.substring(left), k));
        return maxLen;
    }

    //[424].替换后的最长重复字符
    public static int characterReplacement(String s, int k) {
        int left = 0, right = 0, maxCount = 0;
        int[] count = new int[26];
        while (right < s.length()) {
            char ch = s.charAt(right);
            count[ch - 'A']++;
            maxCount = Math.max(maxCount, count[ch - 'A']);

            right++;

            //只关心最长的窗口，即使是窗口缩小，也会维持原来最长的大小
            if (right - left > maxCount + k) {
                count[s.charAt(left) - 'A']--;
                left++;
            }
        }
        return s.length() - left;
    }

    //[438].找到字符串中所有字母异位词
    public static List<Integer> findAnagrams(String s, String p) {
        Map<Character, Integer> need = new HashMap<>();
        for (char ch : p.toCharArray()) {
            need.put(ch, need.getOrDefault(ch, 0) + 1);
        }
        Map<Character, Integer> window = new HashMap<>();

        int left = 0, valid = 0, right = 0;
        List<Integer> res = new ArrayList<>();
        while (right < s.length()) {
            char c = s.charAt(right);
            right++;

            if (need.containsKey(c)) {
                window.put(c, window.getOrDefault(c, 0) + 1);
                if (need.get(c).equals(window.get(c))) {
                    valid++;
                }
            }

            while (right - left >= p.length()) {
                if (need.size() == valid) {
                    res.add(left);
                }
                char l = s.charAt(left);
                left++;

                if (need.containsKey(l)) {
                    if (need.get(l).equals(window.get(l))) {
                        valid--;
                    }
                    window.put(l, window.getOrDefault(l, 0) - 1);
                }
            }
        }
        return res;
    }

    //[443].压缩字符串
    public static int compress(char[] chars) {
        int n = chars.length;
        int left = 0;
        int right = 0;
        int write = 0;
        for (; right <= n; right++) {
            if (right == n || chars[left] != chars[right]) {
                chars[write++] = chars[left];
                if (right - left > 1) {
                    char[] all = String.valueOf(right - left).toCharArray();
                    for (char ch : all) {
                        chars[write++] = ch;
                    }
                }
                left = right;
            }
        }
        return write;
    }

    //[480].滑动窗口中位数
    public static class Solution480 {
        PriorityQueue<Long> small;
        PriorityQueue<Long> large;

        public double[] medianSlidingWindow(int[] nums, int k) {
            small = new PriorityQueue<>((o1, o2) -> {
                if (o2 > o1) {
                    return 1;
                } else if (o2 < o1) {
                    return -1;
                } else {
                    return 0;
                }
            });
            large = new PriorityQueue<>();
            double[] res = new double[nums.length - k + 1];
            int left = 0, right = 0, index = 0;
            while (right < nums.length) {
                int num = nums[right];
                right++;
                addNum(num);

                if (right - left >= k) {
                    res[index++] = median();
                    removeNum(nums[left]);
                    left++;
                }
            }
            return res;
        }

        private void addNum(long num) {
            if (small.size() >= large.size()) {
                small.offer(num);
                large.offer(small.poll());
            } else {
                large.offer(num);
                small.offer(large.poll());
            }
        }

        private void removeNum(long num) {
            if (small.contains(num)) {
                small.remove(num);
                if (large.size() > small.size()) {
                    small.offer(large.poll());
                }
            } else {
                large.remove(num);
                if (small.size() > large.size()) {
                    large.offer(small.poll());
                }
            }
        }

        private double median() {
            if (small.size() > large.size()) {
                return small.peek();
            } else if (small.size() < large.size()) {
                return large.peek();
            } else {
                return small.isEmpty() ? 0 : small.peek() / 2.0d + large.peek() / 2.0d;
            }
        }
    }

    //[567].字符串的排列
    public static boolean checkInclusion(String s1, String s2) {
        int[] need = new int[26];
        Set<Character> needSet = new HashSet<>();
        for (char ch : s1.toCharArray()) {
            need[ch - 'a']++;
            needSet.add(ch);
        }
        int[] window = new int[26];
        int left = 0, right = 0, valid = 0;
        while (right < s2.length()) {
            char ch = s2.charAt(right);
            right++;

            if (needSet.contains(ch)) {
                window[ch - 'a']++;
                if (need[ch - 'a'] == window[ch - 'a']) {
                    valid++;
                }
            }

            //缩窗口
            while (right - left >= s1.length()) {
                if (valid == needSet.size()) {
                    return true;
                }
                char leftCh = s2.charAt(left);
                left++;

                if (needSet.contains(leftCh)) {
                    if (window[leftCh - 'a'] == need[leftCh - 'a']) {
                        valid--;
                    }
                    window[leftCh - 'a']--;
                }
            }
        }
        return false;
    }

    //[581].最短无需连续子数组
    public static int findUnsortedSubarray(int[] nums) {
        int n = nums.length;
        if (n == 1) return 0;
        int l_max = nums[0], right = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] >= l_max) {
                l_max = nums[i];
            } else {
                //更新比前面小的的索引，直到遍历完为止，后面有序部分不会更新。
                right = i;
            }
        }

        int r_min = nums[n - 1], left = n - 1;
        for (int i = n - 1; i >= 0; i--) {
            if (nums[i] <= r_min) {
                r_min = nums[i];
            } else {
                //更新比后面大的的索引，直到遍历完为止，前面有序部分不会更新。
                left = i;
            }
        }
        return right - left > 0 ? right - left + 1 : 0;
    }

    public static void main(String[] args) {
//        [3].无重复子串的最长子串
//        System.out.println(lengthOfLongestSubstring("dvdf"));
//        System.out.println(lengthOfLongestSubstring("pwwkew"));
//        System.out.println(lengthOfLongestSubstring("tmmzuxt"));
//        System.out.println(lengthOfLongestSubstring("tmmzzuxt"));
//        System.out.println(lengthOfLongestSubstring("tmmzuuzt"));
//        System.out.println(lengthOfLongestSubstring("tmmzutt"));
//        System.out.println(lengthOfLongestSubstring("tmmzuzt"));
//        System.out.println(lengthOfLongestSubstring("tmmzuzuzt"));
//        System.out.println(lengthOfLongestSubstring("abcabcbb"));
//        System.out.println(lengthOfLongestSubstring("bbbbb"));
//        System.out.println(lengthOfLongestSubstring("abcabbcad"));
//        System.out.println(lengthOfLongestSubstring("aa"));
//        System.out.println(lengthOfLongestSubstring("pwwwwkeeew"));
//
//        [76].最小覆盖子串
//        System.out.println(minWindow("ADOBECODEBANC", "ABC"));
//        System.out.println(minWindow("a", "c"));
//
//        [209].长度最小的子数组
//        System.out.println(minSubArrayLen(7, new int[]{2, 3, 1, 2, 4, 3}));
//        System.out.println(minSubArrayLen(7, new int[]{2, 3, 9, 2, 4, 3}));
//        System.out.println(minSubArrayLen(7, new int[]{}));
//
//        [239].滑动窗口最大值
//        System.out.println(Arrays.toString(maxSlidingWindow(new int[]{1, 3, -1, -3, 5, 3, 6, 7}, 3)));
//        System.out.println(Arrays.toString(maxSlidingWindow(new int[]{1}, 1)));
//
//        [283].移动零
//        moveZeroes(new int[]{0,1,2,3});
//        moveZeroes(new int[]{0,1,0,3,12});
//        moveZeroes(new int[]{1});
//
//        [287].寻找重复数
//        System.out.println(findDuplicateV2(new int[]{1, 3, 4, 2, 2}));
//
//        [395].至少有k个重复字符的最长子串
//        System.out.println(longestSubstring("ababbcdd", 2));
//        System.out.println(longestSubstring("aaabb", 3));
//
//        [424].替换后的最长重复字符
//        System.out.println(characterReplacement("AABABBA", 1));
//
//        [438].找到字符串中所有字母异位词
//        System.out.println(findAnagrams("abab", "ab"));
//        System.out.println(findAnagrams("cbaebabacd", "abc"));
//
//        [443].压缩字符串
//        System.out.println(compress(new char[]{'a', 'a', 'b', 'b', 'c', 'c', 'c'}));
//        System.out.println(compress(new char[]{'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'}));
//        System.out.println(compress(new char[]{'a'}));
//        System.out.println(compress(new char[]{'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c'}));
//
//        [480].滑动窗口中位数
//        Solution480 solution480 = new Solution480();
//        System.out.println(Arrays.toString(solution480.medianSlidingWindow(new int[]{1, 3, -1, -3, 5, 3, 6, 7}, 4)));
//        System.out.println(Arrays.toString(solution480.medianSlidingWindow(new int[]{-2147483648, -2147483648, 2147483647, -2147483648, -2147483648, -2147483648, 2147483647, 2147483647, 2147483647, 2147483647, -2147483648, 2147483647, -2147483648}, 3)));
//        System.out.println(Arrays.toString(solut ion480.medianSlidingWindow(new int[]{1}, 1)));
//
//        [556].下一个更大元素III
//        System.out.println(nextGreaterElement(14782c));
//
//        [567].字符串的排列
//        System.out.println(checkInclusion("ab", "eidboaoo"));;
//        System.out.println(checkInclusion("ab", "eidbaooo"));;
//
//        [581].最短无需连续子数组
//        System.out.println(findUnsortedSubarray(new int[]{2,6,4,8,10,9,15}));
//        System.out.println(findUnsortedSubarray(new int[]{1,2,3,4,5}));
//
    }
}
