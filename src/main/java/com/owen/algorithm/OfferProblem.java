package com.owen.algorithm;


import com.owen.algorithm.LinkList.ListNode;

public class OfferProblem {

    //[剑指 offer 03].数组中重复的数字
    public static int findRepeatNumber(int[] nums) {

        for (int i = 0; i< nums.length; i++) {
            while (nums[i] != i) {

                //实际位置上的数与当前的数相等就是有重复的数了
                if (nums[nums[i]] == nums[i]) {
                    return nums[i];
                }

                //使得最终nums[i] == i
                int temp = nums[nums[i]];
                nums[nums[i]] = nums[i];
                nums[i] = temp;
            }
        }
        return -1;
    }

    public static String replaceSpace(String s) {
        if (s == null || s.length() == 0) return s;

        StringBuilder sb = new StringBuilder();
        for (char ch : s.toCharArray()) {
            if (ch == ' ') {
               sb.append("%20");
            } else {
                sb.append(ch);
            }
        }
        return sb.toString();
    }


    public static int[] reversePrint(ListNode head) {
        if (head == null) return new int[0];

        int length = 0;
        ListNode cur = head;
        while (cur != null) {
            length++;
            cur = cur.next;
        }

        int[] res = new int[length];
        cur = head;
        while (cur != null) {

            res[--length] = cur.val;
            cur = cur.next;
        }
        return res;
    }

    public static void main(String[] args) {
        System.out.println(findRepeatNumber(new int[] {2, 3, 1, 0, 2, 5, 3}));
    }
}
