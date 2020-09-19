package com.owen.algorithm;

/**
 * Created by OKONG on 2020/9/13.
 */
public class LinkList {
    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
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

    //19.删除链表的倒数第N个节点（双指针）
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

    //21.合并两个有序列表
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

    //25.K个一组翻转链表
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

    //61.旋转链表（双指针）
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
        while(fast != null && fast.next != null) {
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
        while(fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) break;
        }
        //说明无环
        if(fast == null || fast.next == null) return null;
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

    //206. 反转链表迭代
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

    //206.反转链表递归
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

    public static void main(String[] args) {
//        [2]两数相加
//        ListNode f = new ListNode(2);
//        ListNode s = new ListNode(4);
//        ListNode t = new ListNode(3);
//        f.next = s;
//        s.next = t;
//        ListNode f1 = new ListNode(5);
//        ListNode s1 = new ListNode(6);
//        ListNode t1 = new ListNode(4);
//        f1.next = s1;
//        s1.next = t1;
//        ListNode result = addTwoNumbers(f, f1);

//        ListNode z = new ListNode(0);
//        ListNode f = new ListNode(1);
//        ListNode s = new ListNode(2);
//        ListNode t = new ListNode(3);
//        ListNode four = new ListNode(4);
//        ListNode five = new ListNode(5);
//        ListNode six = new ListNode(6);
//        ListNode seven = new ListNode(7);
//        ListNode eight = new ListNode(8);
//        ListNode nine = new ListNode(9);
//
//        z.next = f;
//        f.next = s;
//        f.next = s;
//        s.next = t;
//        t.next = four;
//        four.next = five;
//        five.next = six;
//        six.next = seven;
//        seven.next = eight;
//        eight.next = nine;
//

//        19.删除链表的倒数第N个节点
//        ListNode result = removeNthFromEnd(f, 1);
//
//        25.K个一组翻转链表
//        ListNode result = reverseKGroup(f, 3);
//
//        61.旋转链表
//        ListNode result = rotateRight(null, 4);
//
//        206.反转链表递归
//        ListNode result = reverseList(f);
//
//        21.合并两个有序列表
//        ListNode f = new ListNode(1);
//        ListNode s = new ListNode(2);
//        ListNode t = new ListNode(4);
//        f.next = s;
//        s.next = t;
//        ListNode f1 = new ListNode(1);
//        ListNode s1 = new ListNode(3);
//        ListNode t1 = new ListNode(4);
//        f1.next = s1;
//        s1.next = t1;
//        ListNode result = mergeTwoLists(f, f1);
//
//        24.两两交换链表中的节点
//        ListNode f = new ListNode(1);
//        ListNode s = new ListNode(2);
//        ListNode t = new ListNode(3);
//        ListNode four = new ListNode(4);
//        ListNode five = new ListNode(5);
//        f.next = s;
//        s.next = t;
//        t.next = four;
//        four.next = five;
//        ListNode result = swapPairs(f);
//
//        [86].分隔链表
//        ListNode n1 = new ListNode(1);
//        ListNode n2= new ListNode(4);
//        ListNode n3 = new ListNode(3);
//        ListNode n4 = new ListNode(2);
//        ListNode n5 = new ListNode(5);
//        ListNode n6 = new ListNode(2);
//        n1.next = n2;
//        n2.next = n3;
//        n3.next = n4;
//        n4.next = n5;
//        n5.next = n6;
//        partition(n1, 3);
//        ListNode n1 = new ListNode(1);
//        ListNode n2 = new ListNode(2);
//        ListNode n3 = new ListNode(4);
//        ListNode n4 = new ListNode(5);
//        ListNode n5 = new ListNode(6);
//        ListNode n6 = new ListNode(7);
//        n1.next = n2;
//        n2.next = n3;
//        n3.next = n4;
//        n4.next = n5;
//        n5.next = n6;
//        partition(n6, 8);

//        92.反转列表
//        ListNode f = new ListNode(1);
//        ListNode s = new ListNode(2);
//        ListNode t = new ListNode(3);
//        ListNode four = new ListNode(4);
//        ListNode five = new ListNode(5);
//        f.next = s;
//        s.next = t;
//        t.next = four;
//        four.next = five;
//
//        ListNode a = reverseBetween(f, 1, 1);
//        ListNode b = reverseBetween(f, 2, 4);

        ListNode f = new ListNode(1);
        ListNode s = new ListNode(2);
        ListNode t = new ListNode(3);
        ListNode four = new ListNode(4);
        ListNode five = new ListNode(5);
//        f.next = s;
//        s.next = t;
//        t.next = four;
//        four.next = five;
        reorderList(f);
        System.out.println();

        ListNode result = detectCycle(f);
        System.out.println();
    }
}