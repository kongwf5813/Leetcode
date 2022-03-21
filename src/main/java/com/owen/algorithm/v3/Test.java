package com.owen.algorithm.v3;

import java.util.HashSet;
import java.util.Set;
import java.util.Stack;

import static com.owen.algorithm.v3.AllOfThem.TreeNode;

public class Test {


    public static void main(String[] args) {
        System.out.println(new Test().openLock(new String[]{"5557", "5553", "5575", "5535", "5755", "5355", "7555", "3555", "6655", "6455", "4655", "4455", "5665", "5445", "5645", "5465", "5566", "5544", "5564", "5546", "6565", "4545", "6545", "4565", "5656", "5454", "5654", "5456", "6556", "4554", "4556", "6554"}, "5555"));
    }

    public int openLock(String[] deadends, String target) {
        if (target.equals("0000")) return 0;
        Set<String> deads = new HashSet<>();
        for (String deadend : deadends) deads.add(deadend);
        if (deads.contains("0000")) return -1;

        Set<String> q1 = new HashSet<>();
        q1.add("0000");
        Set<String> q2 = new HashSet<>();
        q2.add(target);

        Set<String> visited = new HashSet<>();
        int res = 0;
        while (!q1.isEmpty()) {
            Set<String> temp = new HashSet<>();
            for (String str : q1) {
                if(deads.contains(str)) continue;
                if (q2.contains(str)) return res;

                visited.add(str);
                for (int i = 0; i < str.length(); i++) {
                    String plusOne = plus(str, i);
                    if (!visited.contains(plusOne)) {
                        temp.add(plusOne);
                    }
                    String minusOne = minus(str, i);
                    if (!visited.contains(minusOne)) {
                        temp.add(minusOne);
                    }
                }
            }
            res++;
            q1 = q2;
            q2 = temp;
        }
        return -1;
    }

    private String plus(String str, int i) {
        char[] arr = str.toCharArray();
        if (arr[i] == '9') arr[i] = '0';
        else arr[i] = (char) (arr[i] + 1);
        return String.valueOf(arr);
    }

    private String minus(String str, int i) {
        char[] arr = str.toCharArray();
        if (arr[i] == '0') arr[i] = '9';
        else arr[i] = (char) (arr[i] - 1);
        return String.valueOf(arr);
    }
}