package com.owen.algorithm.v3;

import java.util.*;

public class Test {

    public static void main(String[] args) {
//        System.out.println(verifyPostorder(new int[] {1,3,2,5,6,7,4}));
//        getLeastNumbers(new int[]{3, 2, 1}, 1);
    }

    public static boolean verifyPostorder(int[] postorder) {
        Stack<Integer> stack = new Stack<>();

        int parent = Integer.MAX_VALUE;
        for (int i = postorder.length - 1; i >= 0; i--) {
            int cur = postorder[i];
            while (!stack.isEmpty() && cur < stack.peek()) {
                parent = stack.pop();
            }
            if (parent < cur) {
                return false;
            }
            stack.push(cur);
        }
        return true;
    }

    public static int[] getLeastNumbers(int[] arr, int k) {
        if (k >= arr.length) return arr;
        return quickSort(arr, 0, arr.length - 1, k);
    }

    private static int[] quickSort(int[] arr, int l, int r, int k) {
        int i = l, j = r;
        int pivot = arr[l];
        while (i < j) {
            while (i < j && arr[r] >= pivot) j--;
            arr[i] = arr[j];
            while (i < j && arr[l] <= pivot) i++;
            arr[j] = arr[i];
        }
        arr[i] = pivot;

        if (i < k) return quickSort(arr, i + 1, r, k);
        else if (i > k) return quickSort(arr, l, i - 1, k);
        return Arrays.copyOf(arr, k);
    }
}
