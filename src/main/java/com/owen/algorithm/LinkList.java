package com.owen.algorithm;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Stack;

/**
 * Created by OKONG on 2020/9/13.
 */
public class LinkList {
    public static class ListNode {
        public int val;
        public ListNode next;

        public ListNode(int x) {
            val = x;
        }
    }

    //[2]两数相加
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode output = new ListNode(0);
        ListNode prev = output;
        //进位标志
        int carry = 0;
        //只要下一位有数据就前进
        while (l1 != null || l2 != null) {
            int value = 0;
            if (l1 != null) {
                value += l1.val;
                l1 = l1.next;
            }

            if (l2 != null) {
                value += l2.val;
                l2 = l2.next;
            }
            value += carry;
            //prev.next才是当前节点
            prev.next = new ListNode(value % 10);
            prev = prev.next;

            //下一位的进位数据
            carry = value / 10;
        }
        if (carry != 0) {
            prev.next = new ListNode(carry);
        }
        //第一个节点直接被忽略掉
        return output.next;
    }

    //[19].删除链表的倒数第N个节点（双指针）
    public static ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode fast = head, slow = head, pre = null;
        int i = 0;
        while (i++ < n) {
            fast = fast.next;
        }
        while (fast != null) {
            pre = slow;
            fast = fast.next;
            slow = slow.next;
        }

        if (pre == null) {
            pre = slow.next;
            return pre;
        } else {
            pre.next = slow.next;
            return head;
        }
    }

    //[21].合并两个有序列表
    public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode prehead = new ListNode(-1);
        ListNode pre = prehead;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                pre.next = l1;
                l1 = l1.next;
            } else {
                pre.next = l2;
                l2 = l2.next;
            }
            pre = pre.next;
        }
        pre.next = l1 == null ? l2 : l1;
        return prehead.next;
    }

    //[23].合并K个升序链表
    public static ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> queue = new PriorityQueue<>(Comparator.comparingInt(o -> o.val));
        for (int i = 0; i < lists.length; i++) {
            if (lists[i] != null) {
                queue.offer(lists[i]);
            }
        }

        ListNode dummyHead = new ListNode(-1);
        ListNode p = dummyHead;
        while (!queue.isEmpty()) {
            ListNode min = queue.poll();
            p.next = min;

            p = p.next;
            if (min.next != null) {
                queue.offer(min.next);
            }
        }
        return dummyHead.next;
    }

    //[24].两两交换链表中的节点
    public static ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode next = head.next;
        ListNode last = swapPairs(next.next);
        next.next = head;
        head.next = last;
        return next;
    }

    //[25].K个一组翻转链表
    public static ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) return null;
        ListNode start = head, end = head;
        for (int i = 0; i < k; i++) {
            //k不足返回的是头
            if (end == null) return head;
            end = end.next;
        }

        ListNode newHead = reverse(start, end);
        start.next = reverseKGroup(end, k);
        return newHead;
    }

    //[61].旋转链表（双指针）
    public static ListNode rotateRight(ListNode head, int k) {
        ListNode cur = head, last = null;
        int n = 0;
        while (cur != null) {
            last = cur;
            cur = cur.next;
            n++;
        }
        int realK;
        if (n == 1 || n == 0 || (realK = k % n) == 0) {
            return head;
        }

        //双指针
        ListNode fast = head, slow = head, start = null;
        int i = 0;
        while (i++ < realK) {
            fast = fast.next;
        }

        while (fast != null) {
            start = slow;
            fast = fast.next;
            slow = slow.next;
        }
        start.next = null;
        last.next = head;
        return slow;
    }

    //[82].删除排序链表中的重复元素II
    public static ListNode deleteDuplicates(ListNode head) {
        ListNode dummyHead = new ListNode(-1);
        dummyHead.next = head;
        ListNode cur = dummyHead;
        while (cur.next != null && cur.next.next != null) {
           if (cur.next.val == cur.next.next.val) {
               ListNode temp = cur.next;
               while (temp != null && temp.next != null && temp.val == temp.next.val) {
                   temp = temp.next;
               }
               cur.next = temp.next;
           } else {
               cur = cur.next;
           }
        }
        return dummyHead.next;
    }

    //[86].分隔链表
    public static ListNode partition(ListNode head, int x) {
        ListNode dummyMinHead = new ListNode(Integer.MIN_VALUE), min = dummyMinHead, dummyMaxHead = null, cur = head, preMax = null;
        while (cur != null) {
            ListNode next = cur.next;
            if (cur.val < x) {
                if (min == dummyMinHead) {
                    //刚开始就给小链表
                    dummyMinHead.next = cur;
                    //min通过这个指针移动
                    min = cur;
                } else {
                    min.next = cur;
                    min = min.next;
                }
                //max通过这个指针移动，将链表粘起来
                if (preMax != null) {
                    preMax.next = next;
                }
            } else {
                //如果有大链表，则给头
                if (dummyMaxHead == null) {
                    dummyMaxHead = cur;
                }
                //max通过这个指针移动
                preMax = cur;
            }
            cur = next;
        }

        if (dummyMaxHead != null) {
            //粘到小链表尾部
            min.next = dummyMaxHead;
        }
        return dummyMinHead.next;
    }

    //[92].反转链表II
    public static ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;
        ListNode s = dummyHead, e = dummyHead.next;
        int count = 0;
        while (count++ < m - 1) {
            s = s.next;
            e = e.next;
        }
        while (count++ < n) {
            ListNode x = e.next;
            e.next = x.next;
            x.next = s.next;
            s.next = x;
        }
        return dummyHead.next;
    }

    //[141]环形链表
    public static boolean hasCycle(ListNode head) {
        if (head == null) return false;
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) return true;
        }
        return false;
    }

    //[142]环形链表II
    public static ListNode detectCycle(ListNode head) {
        if (head == null) return null;
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) break;
        }
        //说明无环
        if (fast == null || fast.next == null) return null;
        slow = head;
        while (slow != fast) {
            slow = slow.next;
            fast = fast.next;
        }
        return slow;
    }

    //[143].重排链表
    public static void reorderList(ListNode head) {
        if (head == null) return;
        ListNode slow = head, fast = head, mid = null;
        while (fast != null && fast.next != null) {
            mid = slow;
            fast = fast.next.next;
            slow = slow.next;
        }
        if (fast != null) {
            mid = slow;
        }
        ListNode q = reverseList(mid.next);
        mid.next = null;
        ListNode p = head;
        while (q != null) {
            ListNode qN = q.next;
            ListNode pN = p.next;
            q.next = pN;
            p.next = q;
            q = qN;
            p = pN;
        }
    }

    //[147].对链表进行插入排序
    public static ListNode insertionSortList(ListNode head) {
        ListNode dummyHead = new ListNode(Integer.MIN_VALUE);
        ListNode f = head;
        while (f != null) {
            ListNode s = dummyHead;
            while (s.val < f.val) {
                if (s.next != null) {
                    if (s.next.val > f.val) {
                        break;
                    }
                    s = s.next;
                } else {
                    break;
                }
            }
            ListNode fastNext = f.next;
            ListNode slowNext = s.next;
            f.next = slowNext;
            s.next = f;
            f = fastNext;
        }
        return dummyHead.next;
    }

    //[148].排序链表
    public static ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode slow = head, fast = head, pre = null;
        while (fast != null && fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        pre.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(slow);
        return merge(left, right);
    }

    private static ListNode merge(ListNode first, ListNode second) {
        ListNode dummyHead = new ListNode(-1);
        ListNode p = first, q = second, cur = dummyHead;
        while (p != null && q != null) {
            if (p.val < q.val) {
                cur.next = p;
                p = p.next;
            } else {
                cur.next = q;
                q = q.next;
            }
            cur = cur.next;
        }

        if (p != null) {
            cur.next = p;
        }
        if (q != null) {
            cur.next = q;
        }
        return dummyHead.next;
    }

    //[203].移除链表元素
    public static ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        ListNode last = removeElements(head.next, val);
        head.next = last;
        if (head.val == val) {
            return head.next;
        } else {
            return head;
        }
    }

    //[206]. 反转链表迭代
    public static ListNode reverse(ListNode start, ListNode end) {
        //注意一定要三个节点，否则可能死循环
        ListNode pre = null, cur = start, next;
        while (cur != end) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        //返回的是遍历到最后的节点
        return pre;
    }

    //[206].反转链表递归
    public static ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode last = reverseList(head.next);
        head.next.next = head;
        //断链呀
        head.next = null;
        return last;
    }

    //[234]回文链表
    private static ListNode left;

    public static boolean isPalindrome(ListNode head) {
        left = head;
        return traverse(head);
    }

    //递归判断链表是否为回文
    private static boolean traverse(ListNode right) {
        if (right == null) return true;
        //right后面的是否为回文
        boolean res = traverse(right.next);
        res = res && left.val == right.val;
        left = left.next;
        return res;
    }

    //[237].删除链表中的节点
    public static void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    //[328].奇偶链表
    public static ListNode oddEvenList(ListNode head) {
        if (head == null) return null;
        ListNode odd = head, even = head.next, x = even;
        while (even != null && even.next != null) {
            odd.next = even.next;
            even.next = even.next.next;
            odd = odd.next;
            even = even.next;
        }
        odd.next = x;
        return head;
    }

    //[382].链表随机节点
    class Solution {

        ListNode head;

        /**
         * @param head The linked list's head.
         *             Note that the head is guaranteed to be not null, so it contains at least one node.
         */
        public Solution(ListNode head) {
            this.head = head;
        }

        /**
         * Returns a random node's value.
         */
        public int getRandom() {
            if (head == null) return -1;

            Random random = new Random();
            ListNode cur = head;
            int res = cur.val;
            int count = 1;

            cur = cur.next;
            while (cur != null) {
                count++;

                int index = random.nextInt(count);
                //取第一个即可
                if (index == 0) {
                    res = cur.val;
                }
                cur = cur.next;
            }
            return res;
        }
    }

    //[430].扁平化多级双向链表
    class Solution430 {
        class Node {
            public int val;
            public Node prev;
            public Node next;
            public Node child;
        }

        public Node flatten(Node head) {
            helper(head);
            return head;
        }

        //返回的是最后一个节点
        public Node helper(Node head) {
            if (head == null) return null;

            Node finalNode = null;
            while (head != null) {
                //遍历父亲节点， 发现有child不为空，进入详细递归，拼接好，需要返回最后一个节点，所以用finalNode表示。
                if (head.child != null) {
                    Node currentParent = head;
                    Node next = currentParent.next;

                    //child转变成下一个next节点
                    head.next = head.child;
                    head.child.prev = head;

                    //返回最后一个节点用于拼接
                    Node lastNode = helper(head.child);
                    head.child = null;

                    //把拉平的节点拼接的原来的next节点
                    if (next != null) {
                        lastNode.next = next;
                        next.prev  = lastNode;
                    }
                }

                finalNode = head;
                head = head.next;
            }
            return finalNode;
        }
    }

    //[445].两数相加II
    public static ListNode addTwoNumbers2(ListNode l1, ListNode l2) {

        Stack<Integer> stack1 = new Stack<>();
        Stack<Integer> stack2 = new Stack<>();

        while(l1 != null) {
            stack1.push(l1.val);
            l1 = l1.next;
        }

        while(l2 != null) {
            stack2.push(l2.val);
            l2 = l2.next;
        }

        int carry = 0;
        ListNode res = null;
        while(!stack1.isEmpty() || !stack2.isEmpty() || carry != 0) {
            int a = stack1.isEmpty()  ? 0 : stack1.pop();
            int b = stack2.isEmpty() ? 0: stack2.pop();
            int cur = a + b + carry;
            carry = cur / 10;
            cur %= 10;
            ListNode currNode = new ListNode(cur);
            currNode.next = res;
            res = currNode;
        }

        return res;
    }

    //[876].链表的中间结点
    public static ListNode middleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    //[1019].链表中的下一个更大节点
    public static int[] nextLargerNodes(ListNode head) {
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

    public static void main(String[] args) {
//        [2]两数相加
//        ListNode f = new ListNode(2);
//        f.next = new ListNode(4);
//        f.next.next = new ListNode(3);
//
//        ListNode f1 = new ListNode(5);
//        f1.next = new ListNode(6);
//        f1.next.next = new ListNode(4);
//        ListNode result = addTwoNumbers(f, f1);
//
//        ListNode f = new ListNode(1);
//        f.next = new ListNode(2);
//        f.next.next = new ListNode(3);
//        f.next.next.next = new ListNode(4);
//        f.next.next.next.next = new ListNode(5);
//        f.next.next.next.next.next = new ListNode(6);
//        f.next.next.next.next.next.next = new ListNode(7);
//        f.next.next.next.next.next.next.next = new ListNode(8);
//        f.next.next.next.next.next.next.next.next = new ListNode(9);
//
//        [19].删除链表的倒数第N个节点
//        ListNode result = removeNthFromEnd(f, 1);
//
//        [25].K个一组翻转链表
//        ListNode result = reverseKGroup(f, 3);
//
//        [61].旋转链表
//        ListNode result = rotateRight(null, 4);
//
//        [206].反转链表递归
//        ListNode result = reverseList(f);
//
//        [21].合并两个有序列表
//        ListNode f = new ListNode(1);
//        f.next = new ListNode(2);
//        f.next.next = new ListNode(4);
//        ListNode f1 = new ListNode(1);
//        f1.next = new ListNode(3);
//        f1.next.next = new ListNode(4);
//        ListNode result = mergeTwoLists(f, f1);
//
//        [23].合并K个升序链表
//        ListNode f = new ListNode(1);
//        f.next = new ListNode(4);
//        f.next.next = new ListNode(5);
//        ListNode s = new ListNode(1);
//        s.next = new ListNode(3);
//        s.next.next = new ListNode(4);
//        ListNode t = new ListNode(2);
//        t.next = new ListNode(6);
//        ListNode res = mergeKLists(new ListNode[]{f, s, t});
//        ListNode res2 = mergeKLists(new ListNode[] {null});
//
//        [24].两两交换链表中的节点
//        ListNode f = new ListNode(1);
//        f.next = new ListNode(2);
//        f.next.next = new ListNode(3);
//        f.next.next.next = new ListNode(4);
//        f.next.next.next.next = new ListNode(5);
//        ListNode result = swapPairs(f);
//
//        [82].删除排序链表中的重复元素II
//        ListNode n1 = new ListNode(1);
//        n1.next= new ListNode(2);
//        n1.next.next = new ListNode(3);
//        n1.next.next.next = new ListNode(3);
//        n1.next.next.next.next = new ListNode(4);
//        n1.next.next.next.next.next = new ListNode(4);
//        n1.next.next.next.next.next.next = new ListNode(5);
//        ListNode res = deleteDuplicates(n1);
//
//        [86].分隔链表
//        ListNode n1 = new ListNode(1);
//        n1.next= new ListNode(4);
//        n1.next.next = new ListNode(3);
//        n1.next.next.next = new ListNode(2);
//        n1.next.next.next.next = new ListNode(5);
//        n1.next.next.next.next.next = new ListNode(2);
//        partition(n1, 3);
//        ListNode n1 = new ListNode(1);
//        n1.next= new ListNode(2);
//        n1.next.next = new ListNode(4);
//        n1.next.next.next = new ListNode(5);
//        n1.next.next.next.next = new ListNode(6);
//        n1.next.next.next.next.next = new ListNode(7);
//        partition(n6, 8);
//
//        [92].反转列表
//        ListNode f = new ListNode(1);
//        f.next = new ListNode(2);
//        f.next.next = new ListNode(3);
//        f.next.next.next = new ListNode(4);
//        f.next.next.next.next = new ListNode(5);
//
//        ListNode a = reverseBetween(f, 1, 1);
//        ListNode b = reverseBetween(f, 2, 4);

//        [143].重排链表
//        ListNode f = new ListNode(1);
//        f.next = new ListNode(2);
//        f.next.next = new ListNode(3);
//        f.next.next.next = new ListNode(4);
//        f.next.next.next.next = new ListNode(5);
//        reorderList(f);
//
//        [142]环形链表II
//        ListNode result = detectCycle(f);

//        [147].对链表进行插入排序
//        -1->5->3->4->0
//        ListNode f = new ListNode(-1);
//        f.next = new ListNode(5);
//        f.next.next = new ListNode(3);
//        f.next.next.next = new ListNode(4);
//        f.next.next.next.next = new ListNode(0);
//        ListNode result = insertionSortList(f);

//        [148].排序链表
//        ListNode f = new ListNode(-1);
//        f.next = new ListNode(5);
//        f.next.next = new ListNode(3);
//        f.next.next.next = new ListNode(4);
//        f.next.next.next.next = new ListNode(0);
//        f.next.next.next.next.next = new ListNode(7);
//        ListNode result = sortList(f);

//        [328].奇偶链表
//        ListNode head = new ListNode(2);
//        head.next = new ListNode(1);
//        head.next.next = new ListNode(3);
//        head.next.next.next = new ListNode(5);
//        head.next.next.next.next = new ListNode(6);
//        head.next.next.next.next.next = new ListNode(4);
//        head.next.next.next.next.next.next = new ListNode(7);
//        ListNode res = oddEvenList(head);
//
//        [445].两数相加II
//        ListNode f = new ListNode(7);
//        f.next = new ListNode(2);
//        f.next.next = new ListNode(4);
//        f.next.next.next = new ListNode(3);
//
//        ListNode f1 = new ListNode(5);
//        f1.next = new ListNode(6);
//        f1.next.next = new ListNode(4);
//        ListNode result = addTwoNumbers2(f, f1);
//
//        [1019].链表中的下一个更大节点
//        ListNode head = new ListNode(2);
//        head.next = new ListNode(7);
//        head.next.next = new ListNode(4);
//        head.next.next.next = new ListNode(3);
//        head.next.next.next.next = new ListNode(5);
//        System.out.println(Arrays.toString(nextLargerNodes(head)));
    }
}
