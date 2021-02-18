package com.owen.algorithm;

import com.amazonaws.services.dynamodbv2.xspec.S;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Stack;
import java.util.concurrent.atomic.AtomicInteger;

public class StackProblem {
    //[150].逆波兰表达式求值
    public static int evalRPN(String[] tokens) {
        Stack<String> stack = new Stack<>();
        for (String token : tokens) {
            if (token.equals("+") || token.equals("-") || token.equals("*") || token.equals("/")) {
                Integer lastNumber = Integer.parseInt(stack.pop());
                Integer firstNumber = Integer.parseInt(stack.pop());
                if (token.equals("+")) {
                    stack.push("" + (firstNumber + lastNumber));
                } else if (token.equals("-")) {
                    stack.push("" + (firstNumber - lastNumber));
                } else if (token.equals("*")) {
                    stack.push("" + (firstNumber * lastNumber));
                } else {
                    stack.push("" + (firstNumber / lastNumber));
                }
            } else {
                stack.push(token);
            }
        }
        return stack.isEmpty() ? 0 : Integer.parseInt(stack.pop());
    }

    public static class MinStack {

        Stack<Long> stack;
        long min;

        /**
         * initialize your data structure here.
         */
        public MinStack() {
            stack = new Stack<Long>();
        }

        public void push(int x) {
            if (stack.isEmpty()) {
                stack.push(0l);
                min = x;
            } else {
                if (x < min) {
                    stack.push(x - min);
                    min = x;
                } else {
                    stack.push(x - min);
                }
            }
        }

        public void pop() {
            if (!stack.isEmpty()) {
                if (stack.peek() < 0) {
                    min = min - stack.peek();
                }
                stack.pop();
            }
        }

        public int top() {
            if (stack.peek() < 0) {
                return (int)min;
            } else {
                return (int)(min + stack.peek());
            }
        }

        public int getMin() {
            return (int)min;
        }
    }

    //[224].基本计算器
    public static int calculate(String s) {
        return dfsForCalculate(s.trim().toCharArray(), new AtomicInteger());
    }

    private static int dfsForCalculate(char[] chars, AtomicInteger index) {
        Stack<Integer> stack = new Stack<>();
        int num = 0;
        char sign = '+';
        for (; index.get() < chars.length; index.incrementAndGet()) {
            char ch = chars[index.get()];
            if (ch == ' ') {
                continue;
            }

            boolean isDig = '0' <= ch && ch <= '9';
            if (isDig) {
                num = num * 10 + (ch - '0');
            }

            //递归计算
            if (ch == '(') {
                index.incrementAndGet();
                num = dfsForCalculate(chars, index);
            }

            if (!isDig || index.get() == chars.length - 1) {
                switch (sign) {
                    case '+':
                        stack.push(num);
                        break;
                    case '-':
                        stack.push(-num);
                        break;
                }
                sign = ch;
                num = 0;
            }

            //递归结束计算
            if (ch == ')') {
                break;
            }
        }

        int res = 0;
        while (!stack.isEmpty()) {
            res += stack.pop();
        }
        return res;
    }

    //[225].队列实现栈
    class MyStack {

        Queue<Integer> queue;

        /**
         * Initialize your data structure here.
         */
        public MyStack() {
            queue = new LinkedList<>();
        }

        /**
         * Push element x onto stack.
         */
        public void push(int x) {
            int size = queue.size();
            queue.offer(x);
            for (int i = 0; i < size; i++) {
                queue.offer(queue.poll());
            }
        }

        /**
         * Removes the element on top of the stack and returns that element.
         */
        public int pop() {
            return queue.poll();
        }

        /**
         * Get the top element.
         */
        public int top() {
            return queue.peek();
        }

        /**
         * Returns whether the stack is empty.
         */
        public boolean empty() {
            return queue.isEmpty();
        }
    }

    //[227].基本计算器II
    public static int calculate2(String s) {
        char[] arr = s.trim().toCharArray();
        Stack<Integer> stack = new Stack<>();
        int num = 0;
        char preSign = '+';
        for (int i = 0; i < arr.length; i++) {
            char single = arr[i];
            if (single == ' ') {
                continue;
            }
            boolean isDig = '0' <= single && single <= '9';
            if (isDig) {
                num = num * 10 + (single - '0');
            }

            if (!isDig || i == arr.length - 1) {
                switch (preSign) {
                    case '+':
                        stack.push(num);
                        break;
                    case '-':
                        stack.push(-num);
                        break;
                    case '*':
                        stack.push(stack.pop() * num);
                        break;
                    case '/':
                        stack.push(stack.pop() / num);
                        break;
                }
                preSign = single;
                num = 0;
            }
        }

        int res = 0;
        while (!stack.isEmpty()) {
            res += stack.pop();
        }
        return res;
    }

    //[232].用栈实现队列
    class MyQueue {

        Stack<Integer> s1;
        Stack<Integer> s2;

        /**
         * Initialize your data structure here.
         */
        public MyQueue() {
            s1 = new Stack<>();
            s2 = new Stack<>();
        }

        /**
         * Push element x to the back of queue.
         */
        public void push(int x) {
            while (!s1.isEmpty()) {
                s2.push(s1.pop());
            }
            s1.push(x);
            while (!s2.isEmpty()) {
                s1.push(s2.pop());
            }
        }

        /**
         * Removes the element from in front of queue and returns that element.
         */
        public int pop() {
            return s1.pop();
        }

        /**
         * Get the front element.
         */
        public int peek() {
            return s1.peek();
        }

        /**
         * Returns whether the queue is empty.
         */
        public boolean empty() {
            return s1.isEmpty();
        }
    }

    //[316].去除重复字母
    public static String removeDuplicateLetters(String s) {
        if (s == null) return null;
        int[] count = new int[256];
        for (char ch : s.toCharArray()) {
            count[ch]++;
        }
        Stack<Character> stack = new Stack<>();
        boolean[] inStack = new boolean[256];
        for (char ch : s.toCharArray()) {
            count[ch]--;
            if (inStack[ch]) {
                continue;
            }
            while (!stack.isEmpty() && stack.peek() > ch) {
                if (count[stack.peek()] == 0) {
                    break;
                }
                inStack[stack.pop()] = false;
            }

            stack.push(ch);
            inStack[ch] = true;
        }

        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        return sb.reverse().toString();
    }

    //[331].验证二叉树的前序序列化
    public static boolean isValidSerialization(String preorder) {
        String[] pre = preorder.split(",");
        Stack<String> stack = new Stack<>();
        for (String ch : pre) {
            if (ch.equals("#")) {
                while (!stack.isEmpty() && stack.peek().equals("#")) {
                    stack.pop();
                    //##， ###
                    if (stack.isEmpty() || stack.pop().equals("#")) return false;
                }
                stack.push(ch);
            } else {
                stack.push(ch);
            }
        }
        //#要比正常节点数多1，最后剩下的一定是#
        return stack.size() == 1 && stack.peek().equals("#");
    }

    //[388].文件的最长绝对路径
    public static int lengthLongestPath(String input) {
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        String[] dirs = input.split("\n");
        int res = 0;
        for (String dir : dirs) {
            int level = dir.lastIndexOf("\t") + 1;

            //需要回退到上一层
            while (level + 1 < stack.size()) {
                stack.pop();
            }

            //默认把/计数
            int len = stack.peek() + dir.length() - level + 1;
            stack.push(len);

            if (dir.contains(".")) {
                res = Math.max(res, len - 1);
            }
        }
        return res;
    }

    //[394].字符串解码
    public static String decodeString(String s) {
        char[] chars = s.toCharArray();
        Stack<String> strings = new Stack<>();
        Stack<Integer> numbers = new Stack<>();
        int curNum = 0;
        StringBuilder curSB = new StringBuilder();
        for (char ch : chars) {
            if (ch >= '0' && ch <= '9') {
                //继续拼接数字
                curNum = curNum * 10 + ch - '0';
                continue;
            } else if (ch == ']') {
                //出栈操作，计算数字和字符串
                int prevNum = numbers.pop();
                String prevStr = strings.pop();
                StringBuilder temp = new StringBuilder();
                for (int i = 0; i < prevNum; i++) {
                    temp.append(curSB);
                }
                //新当前 = 上一个 + 当前*倍数
                curSB = new StringBuilder(prevStr + temp);
            } else if (ch == '[') {
                //数字和字符都结束了
                numbers.push(curNum);
                strings.push(curSB.toString());

                curNum = 0;
                curSB = new StringBuilder();
            } else {
                //继续拼接字符
                curSB.append(ch);
            }
        }
        return curSB.toString();
    }

    //[402].移掉K位数字
    public static String removeKdigits(String num, int k) {
        if (k == num.length()) return "0";
        Stack<Character> stack = new Stack<>();
        for (char ch : num.toCharArray()) {
            while (!stack.isEmpty() && stack.peek() > ch && k > 0) {
                stack.pop();
                k--;
            }
            //前导0，不添加到栈中
            if (stack.isEmpty() && ch == '0') {
                continue;
            }
            stack.push(ch);
        }
        while (k > 0) {
            stack.pop();
            k--;
        }

        if (stack.isEmpty()) {
            return "0";
        }

        StringBuilder sb = new StringBuilder();
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }
        return sb.reverse().toString();
    }

    //[456].132模式
    public static boolean find132pattern(int[] nums) {
        //1 3 2 4
        if (nums.length < 3) return false;
        Stack<Integer> stack = new Stack<>();
        int aj = Integer.MIN_VALUE;
        for (int i = nums.length - 1; i >= 0; i--) {
            int num = nums[i];

            //此处的num为ak
            if (num < aj) {
                return true;
            }

            //只要发现栈顶比较小，保证了 ak > aj，然后尽量取最大值
            while (!stack.isEmpty() && stack.peek() < num) {
                aj = Math.max(stack.pop(), aj);
            }
            stack.push(num);
        }
        return false;
    }

    //[496].下一个更大元素I
    public static int[] nextGreaterElement(int[] nums1, int[] nums2) {
        //nums1 = [4,1,2], nums2 = [1,3,4,2].
        //输出: [-1,3,-1]
        Map<Integer, Integer> map = new HashMap<>();
        Stack<Integer> stack = new Stack<>();
        for (int i = nums2.length - 1; i >= 0; i--) {
            //从右到左放入最大值
            while (!stack.isEmpty() && nums2[i] >= stack.peek()) {
                stack.pop();
            }
            map.put(nums2[i], stack.isEmpty() ? -1 : stack.peek());
            stack.push(nums2[i]);
        }
        int[] res = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) {
            res[i] = map.get(nums1[i]);
        }
        return res;
    }

    //[503].下一个更大的数II
    public static int[] nextGreaterElements(int[] nums) {
        int size = nums.length;
        int[] res = new int[size];
        Stack<Integer> stack = new Stack<>();
        for (int i = 2 * size - 1; i >= 0; i--) {
            while (!stack.empty() && stack.peek() <= nums[i % size]) {
                stack.pop();
            }
            res[i % size] = stack.empty() ? -1 : stack.peek();

            stack.push(nums[i % size]);
        }
        return res;
    }

    //[556].下一个更大元素III
    public static int nextGreaterElement(int n) {
        Stack<Integer> stack = new Stack<>();
        char[] chars = String.valueOf(n).toCharArray();
        int leftIdx = -1, rightIdx = -1;
        for (int i = chars.length - 1; i >= 0; i--) {
            while (!stack.isEmpty() && chars[i] < chars[stack.peek()]) {
                leftIdx = i;
                rightIdx = stack.pop();
            }

            if (leftIdx != -1) {
                break;
            }
            stack.push(i);
        }

        if (leftIdx == -1) return -1;

        char tmp = chars[leftIdx];
        chars[leftIdx] = chars[rightIdx];
        chars[rightIdx] = tmp;
        Arrays.sort(chars, leftIdx + 1, chars.length);

        long ret = Long.parseLong(new String(chars));
        return ret > Integer.MAX_VALUE ? -1 : (int) ret;
    }


    //[735].行星碰撞
    public static int[] asteroidCollision(int[] asteroids) {
        //[-2, -1, 1, 2]
        //[10, 2, -5]
        //[5, 10, -5]
        Stack<Integer> stack = new Stack<>();
        for (int asteroid : asteroids) {
            boolean needAdd = true;
            while (!stack.isEmpty()) {
                //只有这种情况会爆炸
                if (stack.peek() > 0 && asteroid < 0) {
                    //相消
                    if (stack.peek() < -asteroid) {
                        stack.pop();
                    } else if (stack.peek() == -asteroid) {
                        stack.pop();
                        needAdd = false;
                        break;
                    } else {
                        needAdd = false;
                        break;
                    }
                } else {
                    break;
                }
            }
            if (needAdd) {
                stack.push(asteroid);
            }
        }

        int[] res = new int[stack.size()];
        int index = res.length - 1;
        while (!stack.isEmpty()) {
            res[index--] = stack.pop();
        }
        return res;
    }

    //[739].每日温度
    public static int[] dailyTemperatures(int[] T) {
        int size = T.length;
        int[] res = new int[size];
        Stack<Integer> stack = new Stack<>();
        for (int i = size - 1; i >= 0; i--) {
            while (!stack.isEmpty() && T[stack.peek()] <= T[i]) {
                stack.pop();
            }
            res[i] = stack.isEmpty() ? 0 : stack.peek() - i;
            stack.push(i);
        }
        return res;
    }

    //[895].最大频率栈
    static class FreqStack {
        private int maxFreq = 0;
        private Map<Integer, Stack<Integer>> freqNumbers = new HashMap<>();
        private Map<Integer, Integer> numberFreqs = new HashMap<>();

        public FreqStack() {
        }

        public void push(int x) {
            int feq = numberFreqs.getOrDefault(x, 0);
            feq++;
            numberFreqs.put(x, feq);

            freqNumbers.putIfAbsent(feq, new Stack<>());
            Stack<Integer> stack = freqNumbers.get(feq);
            stack.push(x);

            maxFreq = Math.max(maxFreq, feq);
        }

        public int pop() {
            Stack<Integer> stack = freqNumbers.get(maxFreq);
            int res = stack.pop();
            int freq = numberFreqs.get(res);
            freq--;
            numberFreqs.put(res, freq);
            if (stack.isEmpty()) {
                maxFreq--;
            }
            return res;
        }
    }

    //[1019].链表中的下一个更大节点
    public static int[] nextLargerNodes(LinkList.ListNode head) {
        List<Integer> list = new ArrayList<>();
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }

        int[] res = new int[list.size()];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < list.size(); i++) {
            while (!stack.isEmpty() && list.get(stack.peek()) < list.get(i)) {
                int index = stack.pop();
                res[index] = list.get(i);
            }
            stack.push(i);
        }
        return res;
    }

    //        [895].最大频率栈
//        FreqStack freqStack = new FreqStack();
//        freqStack.push(5);
//        freqStack.push(7);
//        freqStack.push(5);
//        freqStack.push(7);
//        freqStack.push(4);
//        freqStack.push(5);
//        System.out.println(freqStack.pop());
//        System.out.println(freqStack.pop());
//        System.out.println(freqStack.pop());
//        System.out.println(freqStack.pop());

    public static void main(String[] args) {
        MinStack minStack = new MinStack();
        minStack.push(-2);
        minStack.push(0);
        minStack.push(-3);
        System.out.println(minStack.getMin());
        minStack.pop();
        System.out.println(minStack.top());
        System.out.println(minStack.getMin());
//        [224].基本计算器
//        System.out.println(calculate("(1+(4+5+2)-3)"));
//        System.out.println(calculate("(1+(4+5+2)-3)+(6+8)"));
//        System.out.println(calculate("(1+2"));
//
//        [227]基本计算器II
//        System.out.println(calculate2("3+2*2"));
//        System.out.println(calculate2(" 3+5 / 2 "));
//        System.out.println(calculate2(" 3/2 "));
//        System.out.println(calculate2(" 1 "));
//
//        [316].去除重复字母
//        System.out.println(removeDuplicateLetters("cbacdcbc"));
//        System.out.println(removeDuplicateLetters("bcabc"));
//        System.out.println(removeDuplicateLetters(""));
//
//        [331].验证二叉树的前序序列化
//        System.out.println(isValidSerialization("9,#,#,1"));
//        System.out.println(isValidSerialization("9,3,4,#,#,1,#,#,2,#,6,#,#"));
//
//        [388].文件的最长绝对路径
//        System.out.println(lengthLongestPath("dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"));
//        System.out.println(lengthLongestPath("dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"));
//
//        [394].字符串解码
//        System.out.println(decodeString("abc"));
//        System.out.println(decodeString("3[a]2[bc]"));
//        System.out.println(decodeString("3[a2[c]]"));
//        System.out.println(decodeString("2[abc]3[cd]ef"));
//        System.out.println(decodeString("abc3[cd]xyz"));
//
//        [402].移掉K位数字
//        System.out.println(removeKdigits("10200", 1));
//        System.out.println(removeKdigits("1432219", 3));
//        System.out.println(removeKdigits("10", 2));
//        System.out.println(removeKdigits("10016042692165669282207674747", 9));
//
//        [456].132模式
//        System.out.println(find132pattern(new int[]{3,4,5,4}));
//
//        [496].下一个更大元素I
//        System.out.println(Arrays.toString(nextGreaterElement(new int[]{4,1,2}, new int[]{1,3,4,2})));
//        System.out.println(Arrays.toString(nextGreaterElement(new int[]{4,1,2}, new int[]{1,3,2,4,2})));
//        System.out.println(Arrays.toString(nextGreaterElement(new int[]{2,4}, new int[]{1,2,3,4})));
//
//        [503].下一个更大元素II
//        System.out.println(Arrays.toString(nextGreaterElements(new int[]{1,2,1})));
//
//
//        [739].每日温度
//        System.out.println(Arrays.toString(dailyTemperatures(new int[]{73, 74, 75, 71, 69, 72, 76, 73})));
//
//        [735].行星碰撞
//        System.out.println(Arrays.toString(asteroidCollision(new int[]{-2, -1, 1, 2})));
//        System.out.println(Arrays.toString(asteroidCollision(new int[]{10, 2, -5})));
//        System.out.println(Arrays.toString(asteroidCollision(new int[]{5, 10, -5})));
//
//        [1019].链表中的下一个更大节点
//        LinkList.ListNode head = new LinkList.ListNode(2);
//        LinkList.ListNode f = new LinkList.ListNode(7);
//        LinkList.ListNode t = new LinkList.ListNode(4);
//        LinkList.ListNode x = new LinkList.ListNode(3);
//        LinkList.ListNode y = new LinkList.ListNode(5);
//        head.next = f;
//        f.next = t;
//        t.next = x;
//        x.next = y;
//        System.out.println(Arrays.toString(nextLargerNodes(head)));
    }
}
