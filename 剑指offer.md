#### 数组中重复的数字

- **题目**：找出数组中重复的数字。

  在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

- **思路1**：使用`HashSet`来判断是否存在过，如果碰到重复的，直接返回接即可，否则加入 `Set`中。

- **思路2**： 因为数字都在 n-1的范围内，所以可以将某个数 num 放到 index = num 的位置上。因为有重复的数字，所以在摆放的时候肯定会发生冲突，这时候返回冲突的数字即可。

```java
class Solution {
    public int findRepeatNumber(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int i : nums) {
            if (set.contains(i)) {
                return i;
            }
            set.add(i);
        }
        return -1;
    }
}


class Solution {
    public int findRepeatNumber(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] != i) {
                if (nums[i] == nums[nums[i]]) {
                    return nums[i];
                }
                int temp = nums[i];
                nums[i] = nums[temp];
                nums[temp] = temp;
            }
        }
        return -1;
    }
}
```





#### 二维数组中的查找

- **题目**： 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
- **思路**： 最简单粗暴的方法就是两层for循环遍历一遍。但是需要 O(N * N) 的时间。 因为题目中每行每列都是有序的，所以可以从左下角或者右上角的位置开始遍历。比如从右上角开始，如果当前的数字比target大，则表明当前列都别target大，则排除当前列；如果当前数字比target小，则当前行所有的数字都比target小，则可以排除当前行，直到找到结果或者遍历结束。

```java
class Solution {
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        int i = 0;
        int j = matrix[0].length - 1;
        while (i < matrix.length && j >= 0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                j--;
            } else {
                i++;
            }
        }
        return false;
    }
}
```





#### 替换空格

- **题目**： 请实现一个函数，把字符串 `s` 中的每个空格替换成"%20"。
- **思路**： 略。

```java
class Solution {
    public String replaceSpace(String s) {
        if (s == null) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        int len = s.length();
        for (int i = 0; i < len; i++) {
            if (s.charAt(i) == ' ') {
                sb.append("%20");
            } else {
                sb.append(s.charAt(i));
            }
        }
        return sb.toString();
    }
}
```



#### 从尾到头打印链表

- **题目**：输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
- **思路**： 反序打印链表，可以利用栈来实现。递归就是天然的栈结构。

```java
class Solution {
    public int[] reversePrint(ListNode head) {
        Stack<Integer> stack = new Stack<>();
        while (head != null) {
            stack.push(head.val);
            head = head.next;
        }
        int[] result = new int[stack.size()];
        int i = 0;
        while (!stack.isEmpty()) {
            result[i++] = stack.pop();
        }
        return result;
    }
}


class Solution {
    public int[] reversePrint(ListNode head) {
        List<Integer> list = new ArrayList<>();
        reversePrint(head, list);
        int[] result = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = list.get(i);
        }
        return result;
    }

    public void reversePrint(ListNode head, List<Integer> list) {
        if (head == null) {
            return ;
        }
        reversePrint(head.next, list);
        list.add(head.val);
    }
}
```



#### 重建二叉树

- **题目**： 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
- **思路**： 前序遍历的第一个值就是根结点，而在中序遍历中，根结点的位置将二叉树分为左右两颗子树。因此可以根据根结点的位置计算出左子树和右子树结点的个数，将前序遍历和中序遍历数组分成左右子树两半，然后再递归生成二叉树即可。

```java
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTree(preorder, inorder, 0, preorder.length - 1, 0, inorder.length - 1);
    }

    public TreeNode buildTree(int[] preorder, int[] inorder, int preStart, int preEnd, int inStart, int inEnd) {
        if (preStart > preEnd) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preStart]);
        int index = findIndex(inorder, preorder[preStart], inStart, inEnd);
        int lenOfLeft = index - inStart;
        root.left = buildTree(preorder, inorder, preStart + 1, preStart + lenOfLeft, inStart, index - 1);
        root.right = buildTree(preorder, inorder, preStart + lenOfLeft + 1, preEnd, index + 1, inEnd);
        return root;
    }
    public int findIndex(int[] arr, int target, int start, int end) {
        for (int i = start; i <= end; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }
}
```



#### 重建二叉树Ⅱ

- **题目**： 输入某二叉树的后序遍历和中序遍历的结果，请重建该二叉树。
- **思路**： 与上一题类似，只是后序遍历根节点的值在数组的最后一位。

```java
class Solution {
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        return buildTree(inorder, postorder, 0, inorder.length - 1, 0, postorder.length - 1);
    }

    public TreeNode buildTree(int[] inorder, int[] postorder, int inStart, int inEnd, int postStart, int postEnd) {
        if (inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(postorder[postEnd]);
        int index = findIndex(inorder, inStart, inEnd, postorder[postEnd]);
        int lenOfLeft = index - inStart;
        root.left = buildTree(inorder, postorder, inStart, index - 1, postStart, postStart + lenOfLeft - 1);
        root.right = buildTree(inorder, postorder, index + 1, inEnd, postStart + lenOfLeft, postEnd - 1);
        return root;
    }

    public int findIndex(int[] arr, int start, int end, int target) {
        for (int i = start; i <= end; i++) {
            if (arr[i] == target) {
                return i;
            }
        }
        return -1;
    }
}
```



#### 二叉树的下一个节点

- **题目**：给定一个二叉树其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的next指针。
- **思路**： 二叉树中序遍历中某个节点的下一个节点，存在三种情况： 1）该节点存在右子树，则下一个节点就是右子树中最左边的节点； 2）该节点不存在右子树，但是该节点是其父节点的左子节点，这时候父节点就是下一个节点； 3）该节点不存在右子树，并且该节点是其父节点的右子节点，那么下一个节点只能顺着根节点去找，直到找到某个父节点是爷节点的左子节点，或者找不到返回null。  （2和3可以统一处理）

```java
public class Solution {
    public TreeLinkNode GetNext(TreeLinkNode pNode) {
        if (pNode == null) {
            return null;
        }
        if (pNode.right != null) {
            TreeLinkNode cur = pNode.right;
            while (cur.left != null) {
                cur = cur.left;
            }
            return cur;
        }
        TreeLinkNode cur = pNode;
        while (cur.next != null) {
            if (cur.next.left == cur) {
                return cur.next;
            }
            cur = cur.next;
        }
        return null;
    }
}
```

