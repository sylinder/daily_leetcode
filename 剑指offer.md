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



#### 用两个栈实现队列

- **题目**： 用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )
- **思路**： 队列先进先出，栈后进先出，完全相反，怎么才能让两个栈实现队列的功能呢？负负得正。先将数据加入第一个栈中，然后从第一个栈再加入第二个栈中，这样就可以实现先进先出的功能了。但是需要注意的是，从第一个栈加入第二个栈的时机，如果第二个栈里面有数据（先进的），此时加入则会使得第二个栈中原本先进来的数据后面才出。因此，需要在第二个栈为空时才将第一个栈的数据全部加入第二个栈中。

```java
class CQueue {
    private Stack<Integer> stack1;
    private Stack<Integer> stack2;

    public CQueue() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }
    
    public void appendTail(int value) {
        stack1.push(value);
    }
    
    public int deleteHead() {
        if (!stack2.isEmpty()) {
            return stack2.pop();
        }
        while (!stack1.isEmpty()) {
            stack2.push(stack1.pop());
        }
        return stack2.isEmpty() ? -1 : stack2.pop();
    }
}
```



#### 用队列实现栈

- **题目**： 请你仅使用两个队列实现一个后入先出（LIFO）的栈，并支持普通队列的全部四种操作（`push`、`top`、`pop` 和 `empty`）。
- **思路**： 申请两个队列，queue1 保持空的状态，queue2 假设维护着后进先出的顺序，此时push操作只需要加入到queue1中，然后再将queue2中的所有元素都加入queue2中，即可保证此时的queue1中的顺序就是后进先出的顺序。再将queue1和queue2交换，让queue1继续处理push操作，queue2维持后进先出的顺序。而queue2中的顺序，我们一开始就维护好即可。

```java
class MyStack {

    private Queue<Integer> queue1;
    private Queue<Integer> queue2;

    /** Initialize your data structure here. */
    public MyStack() {
        queue1 = new LinkedList<>();
        queue2 = new LinkedList<>();
    }
    
    /** Push element x onto stack. */
    public void push(int x) {
        queue1.add(x);
        while (!queue2.isEmpty()) {
            queue1.add(queue2.poll());
        }
        Queue<Integer> temp = queue1;
        queue1 = queue2;
        queue2 = temp;
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        return empty() ? -1 : queue2.poll();
    }
    
    /** Get the top element. */
    public int top() {
        return empty() ? -1 : queue2.peek();
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
        return queue2.isEmpty();
    }
}
```



#### 包含min函数的栈

- **题目**： 定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。
- **思路**： 在结构内部定义两个栈，dataStack用来正常保存push进来的数据，minStack用来保存当前栈中的最小值。当push的时候，如果minStack为空
- 或者minStack的栈顶元素比push进来的数x要小，则将x 加入到同时加入dataStack和 minStack中，否则将x插入dataStack，而将minStack的栈顶元素再次加入minStack中。pop的时候同时将dataStack和minStack中的元素pop出来即可。

```java
class MinStack {
    private Stack<Integer> dataStack;
    private Stack<Integer> minStack;

    /** initialize your data structure here. */
    public MinStack() {
        dataStack = new Stack<>();
        minStack = new Stack<>();
    }
    
    public void push(int x) {
        if (minStack.isEmpty() || minStack.peek() > x) {
            minStack.push(x);
        } else {
            minStack.push(minStack.peek());
        }
        dataStack.push(x);
    }
    
    public void pop() {
        dataStack.pop();
        minStack.pop();
    }
    
    public int top() {
        return dataStack.peek();
    }
    
    public int min() {
        return minStack.peek();
    }
}
```



#### 栈的压入、弹出序列

- **题目**： 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。
- **思路**： 用两个序列来模拟入栈和出栈的过程。将pushed数组中的元素分别压入栈中，然后判断栈顶元素是否等于popped数组的下标index，如果相等，则将栈顶元素出栈并将index加一。最后判断栈是否为空即可。

```java
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> stack = new Stack<>();
        int pushedIndex = 0;
        int poppedIndex = 0;
        while (pushedIndex < pushed.length) {
            stack.push(pushed[pushedIndex++]);
            while (!stack.isEmpty() && stack.peek() == popped[poppedIndex]) {
                stack.pop();
                poppedIndex++;
            }
        }
        return stack.isEmpty();
    }
}
```



#### 斐波那契数列

- **题目**： 写一个函数，输入 `n` ，求斐波那契（Fibonacci）数列的第 `n` 项（即 `F(N)`）。斐波那契数列的定义如下：

  ```java
  F(0) = 0,   F(1) = 1
  F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
  ```

  斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。答案需要取模 1e9+7（1000000007）。

- **思路**： 如果暴力解决，即 fib = fib (n - 1) + fib (n - 2)，将会严重超时。主要是因为会存在很多很多重复的计算。比如，暴力法在计算 fib(8) 时，会计算 fib(7) 和 fib(6)，而 fib(7) 会计算 fib(6) 和 f(5)……在整个计算的过程当中，f(6)计算了2次，f(5)计算了3次，f(4)计算了5次…… 随着n的增长，时间复杂度将会以指数的方式增长……

  动态规划的解法： fib(n)只与 fib(n - 1) 和 fib(n - 2)有关，而 fib(n - 1) 只与 fib(n - 2) 和 fib(n - 3) 有关……只要自底向上地保存前面计算的两个数，那么计算 fib (n) 只需要简单的加法即可。状态转移方程题目已经给出。

```java
class Solution {
    public int fib(int n) {
        if (n <= 1) {
            return n;
        }
        int first = 0;
        int second = 1;
        int result = 0;
        for (int i = 2; i <= n; i++) {
            result = (first + second) % 1000000007; //题目要求取余，否则会有溢出
            first = second;
            second = result;
        }
        return second;
    }
}
```



#### 青蛙跳台阶问题Ⅰ

- **题目**： 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 `n` 级的台阶总共有多少种跳法。答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
- **思路**： 跳上n级的台阶，可以从 n - 1 和 n - 2 这两级台阶跳上去。即 `F(n) = F(n - 1) + F(n - 2)`。 这其实就是一个斐波那契数列。

```java
class Solution {
    public int numWays(int n) {
        if (n <= 1) {
            return 1; 
        }
        int first = 1;
        int second = 1;
        int result = 0;
        for (int i = 2; i <= n; i++) {
            result = (first + second) % 1000000007;
            first = second;
            second = result;
        }
        return result;
    }
}
```



#### 青蛙跳台阶问题Ⅱ

- **题目**： 一只青蛙一次可以跳上1级台阶，可以跳上2级台阶……也可以跳上n级台阶。求该青蛙跳上一个 `n` 级的台阶总共有多少种跳法。
- 思路： 跳上n级台阶，可以从第 n - 1 级、第 n - 2 级 、…… 、第0级跳上去。因此状态转移方程为： `F(n) = F(n - 1) + F(n - 2) + ... + F(0)` ，然后 `F(n - 1) = F(n - 2) + F(n - 3) + ... + F(0)`， 可以发现，`F(n) = 2F(n - 1) = 4 F(n - 2) = ... `而`F(0) = F(1) = 1`。最终可得F(n) 等于2的(n-1)次方。用位运算表示即： 2 左移 n - 1 位。

```java
public class Solution {
    public int jumpFloorII(int target) {
        return 1 << (target - 1);
    }
}
```



#### 矩形覆盖

- **题目**： 我们可以用2 * 1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2 * 1的小矩形无重叠地覆盖一个2 * n的大矩形，总共有多少种方法？
- **思路**： 对于一个 2 * n 的大矩形，假设完全覆盖的方法有 F(n) 种，而最左边可以有两种放置方式：竖着放和横着放。竖着放的时候，右边还有 n - 1 个空位； 横着放在左上角的时候，左下角也必须横着放一个 2 * 1的小矩形，而此时右边还有 n - 2 个空位。即 `F(n) = F(n - 1) + F(n - 2)` 。这其实也是一个斐波那契数列。

```java
public class Solution {
    public int rectCover(int target) {
        if (target <= 2) {
            return target;
        }
        int first = 1;
        int second = 2;
        int result = 0;
        for (int i = 3; i <= target; i++) {
            result = first + second;
            first = second;
            second = result;
        }
        return result;
    }
}
```

