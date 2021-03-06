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

- **思路**： 如果暴力解决，即 fib = fib (n - 1) + fib (n - 2)，将会严重超时。主要是因为会存在很多很多重复的计算(很多重复的子树）。比如，暴力法在计算 fib(8) 时，会计算 fib(7) 和 fib(6)，而 fib(7) 会计算 fib(6) 和 f(5)……在整个计算的过程当中，f(6)计算了2次，f(5)计算了3次，f(4)计算了5次…… 随着n的增长，时间复杂度将会以指数的方式增长……

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



#### 旋转数组的最小数字

- **题目**： 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。
- **思路**： 有序数组的查找，那肯定是二分了。旋转数组分为前后两部分有序数组，前面那部分大于或等于后面那部分，因此在使用二分法的时候，可以将中间的元素 mid 和数组的最后一个元素high 相比，如果比high大，那么说明mid在前面一半，最小值在mid的右边；如果比high小，那说明最小值在左手边或者是mid自己；如果mid和high相等，那什么可能都有，只能一个一个遍历了。

```java
class Solution {
    public int minArray(int[] numbers) {
        if (numbers == null || numbers.length == 0) {
            return -1;
        }
        int low = 0;
        int high = numbers.length - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (numbers[mid] > numbers[high]) {
                low =  mid + 1;
            } else if (numbers[mid] < numbers[high]) {
                high = mid;
            } else {
                high--;
            }
        }
        return numbers[low];
    }
}
```



#### 矩阵中的路径

- **题目**： 给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

  单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

- **思路**： 回溯法。使用回溯法，首先要搞清楚三件事：1）可以选择的列表。 2）已经选择的列表。 3）结束条件。 对于可以选择的列表，当已经选择了某一个值的时候，此时的选择列表中有上下左右四个选项了（并且选项中未被访问过）； 对于已经选择的列表，那就是找到了word字符串的前面几个，用一个int下标表示即可； 对于结束条件，在board中找到word字符串即可（即已选择的列表中的下标等于word的长度）。 明确了这三点，然后套用回溯法的框架即可，剩下的就交给递归解决了。

```java
void backtrack(selectedPath, selectOptions) {
    if (satisfied) {
        result.add();
        return;
    }
    for (option : selectOptions) {
        choose(option); // selectedPath.add(option)
        backtrack(selectedPath, selectOptions);
        popOption(option); // delete this option and choose another
    }
}
```



```java
class Solution {
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || word == null) {
            return false;
        }
        int row = board.length;
        int col = board[0].length;
        boolean[][] visited = new boolean[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (backtrack(board, i, j, word, 0, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    public boolean backtrack(char[][] board, int i, int j, String word, int pathLen, boolean[][] visited) {
        if (pathLen == word.length()) {
            return true;
        }
        boolean hasPath = false;
        if (i >= 0 && i < board.length && j >= 0 && j < board[0].length && !visited[i][j] && board[i][j] == word.charAt(pathLen)) {
            pathLen++;
            visited[i][j] = true;
            
            hasPath = backtrack(board, i - 1, j, word, pathLen, visited) ||
                      backtrack(board, i + 1, j, word, pathLen, visited) || 
                      backtrack(board, i, j - 1, word, pathLen, visited) || 
                      backtrack(board, i, j + 1, word, pathLen, visited);
            
            pathLen--;
            visited[i][j] = false;
        }
        return hasPath;
    }
}
```



#### 机器人的活动范围

- **题目**： 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？
- **思路**： 回溯法。
  - 可以选择的列表：机器人站在某个方格内，可以往上下左右四个方向走。（如果没访问过的话）
  - 已经选择的列表： 用一个二维布尔数组来表示，true表示访问过，false表示还没访问过。
  - 结束条件或者满足条件：行坐标和列坐标的数位之和不大于k即可。

```java
class Solution {
    private int result = 0;

    public int movingCount(int m, int n, int k) {
        if (m <= 0 || n <= 0) {
            return result;
        }
        boolean[][] visited = new boolean[m][n];
        backtrack(m, n, 0, 0, k, visited);
        return result;
    }

    private void backtrack(int row, int col, int i, int j, int k, boolean[][] visited) {
        if (i >= 0 && i < row && j >= 0 && j < col && !visited[i][j] && getDigit(i) + getDigit(j) <= k) {
            result++;
            visited[i][j] = true;
            backtrack(row, col, i - 1, j, k, visited);
            backtrack(row, col, i + 1, j, k, visited);
            backtrack(row, col, i, j - 1, k, visited);
            backtrack(row, col, i, j + 1, k, visited);
        }
    }

    private int getDigit(int num) {
        int sum = 0;
        while (num > 0) {
            sum += num % 10;
            num /= 10;
        }
        return sum;
    } 
}
```



#### 剪绳子

- **题目**：给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
- **思路**： 设`f(n)`是长度为n的绳子剪成若干段的最大乘积，则状态转移方程为： `f(n) = max(f(i) * f(n-i))`。然后自底向上计算即可。

```java
class Solution {
    public int cuttingRope(int n) {
        if (n < 2) {
            return 0;
        }
        if (n == 2) {
            return 1;
        }
        if (n == 3) {
            return 2;
        }
        int[] dp = new int[n + 1];
        for (int i = 1; i <= 3; i++) {
            dp[i] = i;
        }
        for (int i = 4; i <= n; i++) {
            int max = 0;
            for (int j = 1; j <= i / 2; j++) {
                int temp = dp[j] * dp[i - j];
                if (max < temp) {
                    max = temp;
                }
                dp[i] = max;
            }
        }
        return dp[n];
    }
}
```



#### 二进制中1的个数

- **题目**： 请实现一个函数，输入一个整数（以二进制串形式），输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。
- **思路**： 略。

```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int result = 0;
        while (n != 0) {
            result += n & 1;
            n >>>= 1;
        }
        return result;
    }
}
```



#### 删除链表的节点

- **题目**： 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

  返回删除后的链表的头节点。

- **思路**： 略。

```java
class Solution {
    public ListNode deleteNode(ListNode head, int val) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode cur = dummy;
        while (cur.next != null) {
            if (cur.next.val == val) {
                cur.next = cur.next.next;
                break;
            }
            cur = cur.next;
        }
        return dummy.next;
    }
}
```



#### 调整数组顺序使奇数位于偶数前面

- **题目**： 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
- **思路**： 双指针，前面一个low，后面一个high。low一直往后走，直到遇到偶数； high一直往前走，直到遇到奇数，交换两个值；然后重复上面的过程，直到low遇见high。

```java
class Solution {
    public int[] exchange(int[] nums) {
        if (nums == null || nums.length < 2) {
            return nums;
        }
        int low = 0;
        int high = nums.length - 1;
        while (low < high) {
            while (low < high && nums[low] % 2 == 1) {
                low++;
            }
            while (low < high && nums[high] % 2 == 0) {
                high--;
            }
            swap(nums, low, high);
        }
        return nums;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```



#### 链表中倒数第K个节点

- **题目**： 输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有 `6` 个节点，从头节点开始，它们的值依次是 `1、2、3、4、5、6`。这个链表的倒数第 `3` 个节点是值为 `4` 的节点。
- **思路**： 快慢指针。fast指针先走K步，然后fast指针和slow指针一起走，直到fast指针为空。

```java
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        if (head == null) {
            return null;
        }
        ListNode fast = head;
        ListNode slow = head;
        for (int i = 0; i < k; i++) {
            if (fast == null) {
                return null;
            }
            fast = fast.next;
        }
        while (fast != null) {
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }
}
```



#### 反转链表

- **题目**： 定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
- **思路**： 略。

```java
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode cur = head;
        ListNode pre = null;
        while (cur != null) {
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
}
```



#### 合并两个排序的链表

- **题目**： 输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。
- **思路**： 略。

```java
class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        while (l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                cur.next = l1;
                l1 = l1.next;
            } else {
                cur.next = l2;
                l2 = l2.next;
            }
            cur = cur.next;
        }
        if (l1 != null) {
            cur.next = l1;
        }
        if (l2 != null) {
            cur.next = l2;
        }
        return dummy.next;
    }
}
```





#### 二叉树的镜像

- **题目**： 请完成一个函数，输入一个二叉树，该函数输出它的镜像。
- **思路**： 从根节点开始，每个节点需要做的就是交换左右子树，然后递归处理左右子树即可。

```java
class Solution {
    public TreeNode mirrorTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        mirrorTree(root.left);
        mirrorTree(root.right);
        return root;
    }
}
```



#### 对称的二叉树

- **题目**： 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。
- **思路**： 略。

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isSymmetric(root.left, root.right);
    }

    public boolean isSymmetric(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return isSymmetric(left.left, right.right) && isSymmetric(left.right, right.left);
    }
}
```



#### 数组中出现次数超过一半的数字

- **题目**： 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
- **思路**： 略。

```java
class Solution {
    public int majorityElement(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int result = nums[0];
        int times = 1;
        for (int i = 1; i < nums.length; i++) {
            if (times == 0) {
                result = nums[i];
                times = 1;
            } else if (result == nums[i]) {
                times++;
            } else {
                times--;
            }
        }
        return result;
    }
}
```



#### 树的子结构

- **题目**： 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

  B是A的子结构， 即 A中有出现和B相同的结构和节点值。

- **思路**： 遍历二叉树A，在A中找与B根节点值相等的节点，然后判断这些节点的子树是否包含B即可。

```java
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        boolean result = false;
        if (A != null && B != null) {
            if (A.val == B.val) {
                result = hasSubTree(A, B);
            }
            if (!result) {
                result = isSubStructure(A.left, B);
            }
            if (!result) {
                result = isSubStructure(A.right, B);
            }
        }
        return result;
    }

    private boolean hasSubTree(TreeNode A, TreeNode B) {
        if (B == null) {
            return true;
        }
        if (A == null) {
            return false;
        }
        if (A.val != B.val) {
            return false;
        }
        return hasSubTree(A.left, B.left) && hasSubTree(A.right, B.right);
    }
}
```





#### 从上到下打印二叉树Ⅰ

- **题目**： 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
- **思路**： 二叉树的层次遍历。

```java
class Solution {
    public int[] levelOrder(TreeNode root) {
        if (root == null) {
            return new int[]{};
        }
        List<Integer> list = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode cur = queue.poll();
            list.add(cur.val);
            if (cur.left != null) {
                queue.offer(cur.left);
            }
            if (cur.right != null) {
                queue.offer(cur.right);
            }
        }
        int[] result = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = list.get(i);
        }
        return result;
    }
}
```



#### 从上到下打印二叉树Ⅱ

- **题目**： 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
- **思路**： 略。

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> temp = new ArrayList<>();
            while (size-- > 0) {
                TreeNode cur = queue.poll();
                temp.add(cur.val);
                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            result.add(temp);
        }
        return result;
    }
}
```



#### 从上到下打印二叉树Ⅲ

- **题目**： 请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
- **思路**： 略。

```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        Stack<TreeNode> stack1 = new Stack<>();
        Stack<TreeNode> stack2 = new Stack<>();
        stack1.push(root);
        int level = 1;
        while (!stack1.isEmpty() || !stack2.isEmpty()) {
            List<Integer> temp = new ArrayList<>();
            if (level % 2 == 1) {
                while (!stack1.isEmpty()) {
                    TreeNode cur = stack1.pop();
                    temp.add(cur.val);
                    if (cur.left != null) {
                        stack2.push(cur.left);
                    }
                    if (cur.right != null) {
                        stack2.push(cur.right);
                    }
                }
            } else {
                while (!stack2.isEmpty()) {
                    TreeNode cur = stack2.pop();
                    temp.add(cur.val);
                    if (cur.right != null) {
                        stack1.push(cur.right);
                    }
                    if (cur.left != null) {
                        stack1.push(cur.left);
                    }
                }
            }
            level++;
            result.add(temp);
        }
        return result;
    }
}
```



#### 二叉搜索树的后序遍历序列

- **题目**： 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 `true`，否则返回 `false`。假设输入的数组的任意两个数字都互不相同。
- **思路**： 根据二叉搜索树的特点，在数组中找到第一个比根节点rootValue大的值，然后将数组分成两半，前面一半必须小于rootValue，后面一半必须大于rootValue。前面一半小于rootValue在遍历查找的时候已经比较过了，所以只需要检查后面那一半是否都大于rootValue即可。如果都有小于rootValue的，那就说明这个不是BST的后序遍历，否则，递归调用左右两半即可。

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        if (postorder == null || postorder.length <= 1) {
            return true;
        }
        return postorder(postorder, 0, postorder.length - 1);
    }

    private boolean postorder(int[] nums, int start, int end) {
        if (start >= end) {
            return true;
        }
        int rootValue = nums[end];
        int index = start;
        while (nums[index] < rootValue) {
            index++;
        }
        if (!isValid(nums, index, end - 1, rootValue)) {
            return false;
        }
        return postorder(nums, start, index - 1) && postorder(nums, index, end - 1);
    }

    private boolean isValid(int[] nums, int start, int end, int value) {
        for (int i = start; i <= end; i++) {
            if (nums[i] < value) {
                return false;
            }
        }
        return true;
    }
}
```



#### 二叉树中和为某一值的路劲

- **题目**： 输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。
- **思路**： 因为要求路径和等于某个值，所以从根节点到叶子节点需要记录路径和的值，并且需要记录整条路径的所有节点值。因此对于某个节点来说，需要做的事情是：接收父节点传下来的路径和记录已经走过路径节点值的list，然后将自己的值加入期中。如果此节点是叶子节点并且刚好路径和为target，那么就将其加入结果中。当离开该节点时，需要将其从路径和路径和中删除。

```java
class Solution {
    private List<List<Integer>> result = new LinkedList<>();

    public List<List<Integer>> pathSum(TreeNode root, int target) {
        if (root == null) {
            return result;
        }
        LinkedList<Integer> list = new LinkedList<>();
        pathSum(root, target, list);
        return result;
    }

    public void pathSum(TreeNode root, int target, LinkedList<Integer> list) {
        if (root == null) {
            return ;
        }
        target -= root.val;
        list.addLast(root.val);
        if (root.left == null && root.right == null && target == 0) {
            result.add(new LinkedList(list));
        }
        pathSum(root.left, target, list);
        pathSum(root.right, target, list);
        target += root.val;
        list.removeLast();
    }
}
```



#### 复杂链表的复制

- **题目**： 请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。
- **思路**： 在原链表的每个节点后面插入一个新的copy节点，然后再将每个copy节点的random指针都指向对应的位置（原来节点的random指针的下一个节点），最后将链表分开即可。

```java
class Solution {
    public Node copyRandomList(Node head) {
       if (head == null) {
           return head;
       } 
       Node cur = head;
       while (cur != null) {
           Node next = cur.next;
           cur.next = new Node(cur.val);
           cur.next.next = next;
           cur = next;
       }
       cur = head;
       while (cur != null) {
           Node copy = cur.next;
           if (cur.random != null) {
               copy.random = cur.random.next;
           }
           cur = copy.next;
       }
       cur = head;
       Node result = head.next;
       while (cur != null) {
           Node copy = cur.next;
           cur.next = copy.next;
           if (cur.next != null) {
               copy.next = cur.next.next;
           }
           cur = cur.next;
       }
       return result;
    }
}
```



#### 序列化二叉树

- **题目**： 请实现两个函数，分别用来序列化和反序列化二叉树。
- **思路**： 略。

```java
public class Codec {

    private int index = 0;

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) {
            return "#,";
        }
        String str = root.val + ",";
        str += serialize(root.left);
        str += serialize(root.right);
        return str;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] strs = data.split(",");
        return deserialize(strs);
    }

    private TreeNode deserialize(String[] strs) {
        String str = strs[index++];
        if (str.equals("#")) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(str));
        root.left = deserialize(strs);
        root.right = deserialize(strs);
        return root;
    }
}
```



#### 字符串的排列

- **题目**： 输入一个字符串，打印出该字符串中字符的所有排列。你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。
- **思路**：回溯法。
  - 已经选择的列表：已经选了第几个，可用一个StringBuilder或者list来记录。
  - 可选择的列表：除了已经选的那些。
  - 结束条件：选完了所有字符，即字符串的长度等于sb的长度
  - （注意由重复字符的情况，代码有时间再优化一下）

```java
class Solution {
    private List<String> result = new LinkedList<>();

    public String[] permutation(String s) {
        if (s == null || s.length() == 0) {
            return new String[]{};
        }
        boolean[] visited = new boolean[s.length()];
        Set<String> set = new HashSet<>();
        StringBuilder sb = new StringBuilder();
        backtrack(s, visited, sb, set);
        return listToArray(result);
    }

    private void backtrack(String s, boolean[] visited, StringBuilder sb, Set<String> set) {
        if (sb.length() == s.length()) {
            if (!set.contains(sb.toString())) {
                set.add(sb.toString());
                result.add(sb.toString());
            }
            return ;
        }
        for (int i = 0; i < s.length(); i++) {
            if (visited[i]) {
                continue;
            }
            sb.append(s.charAt(i));
            visited[i] = true;
            backtrack(s, visited, sb, set);
            sb.deleteCharAt(sb.length() - 1);
            visited[i] = false;
        }
    }

    private String[] listToArray(List<String> list) {
        String[] result = new String[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = list.get(i);
        }
        return result;
    }
}
```



#### 最小的k个数

- **题目**： 输入整数数组 `arr` ，找出其中最小的 `k` 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
- **思路**：
  - 堆排序。把所有元素都加到大根堆里面，并维持堆的大小为K，那么最后剩下的k个数就是所要的结果。
  - 快速排序。

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if (arr == null || arr.length == 0 || arr.length < k || k == 0) {
            return new int[] {};
        }
        int[] result = new int[k];
        PriorityQueue<Integer> heap = new PriorityQueue<>((o1, o2) -> o2 - o1);
        for (int i : arr) {
            if (heap.size() < k) {
                heap.add(i);
            } else if (heap.peek() > i) {
                heap.poll();
                heap.add(i);
            }
        }
        for (int i = 0; i < k; i++) {
            result[i] = heap.poll();
        }
        return result;
    }
}
```

```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if (arr == null || arr.length == 0 || k == 0 || k > arr.length) {
            return new int[] {};
        }
        int start = 0;
        int end = arr.length - 1;
        int index = partition(arr, start, end);
        while (index != k - 1) {
            if (index > k - 1) {
                end = index - 1;
                index = partition(arr, start, end);
            } else {
                start = index + 1;
                index = partition(arr, start, end);
            }
        }
        int[] result = new int[k];
        for (int i = 0; i < k; i++) {
            result[i] = arr[i];
        }
        return result;
    }

    private int partition(int[] arr, int low, int high) {
        int pivot = arr[low];
        int i = low;
        int j = high + 1;
        while (true) {
            while (++i < high && arr[i] < pivot);
            while (--j > low && arr[j] > pivot);
            if (i >= j) {
                break;
            }
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
        arr[low] = arr[j];
        arr[j] = pivot;
        return j;
    }
}
```



#### 数据流中的中位数

- **题目**： 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。
- **思路**： 用一个大根堆保存较小的那一半，用另外一个小根堆保存较大的那一半。因此中位数只需要根据奇偶数来选取堆顶元素即可。

```java
class MedianFinder {

    private PriorityQueue<Integer> minHeap;
    private PriorityQueue<Integer> maxHeap;
    private Integer total;
    /** initialize your data structure here. */
    public MedianFinder() {
        minHeap = new PriorityQueue<>();
        maxHeap = new PriorityQueue<>((o1, o2) -> o2 - o1);
        total = 0;
    }
    
    public void addNum(int num) {
        if (total % 2 == 0) {
            maxHeap.add(num);
            minHeap.add(maxHeap.poll());
        } else {
            minHeap.add(num);
            maxHeap.add(minHeap.poll());
        }
        total++;
    }
    
    public double findMedian() {
        if (total == 0) {
            return 0.0;
        }
        if (total % 2 == 1) {
            return minHeap.peek();
        }
        return (minHeap.peek() + maxHeap.peek()) / 2.0;
    }
}
```



#### 连续子数组的最大和

- **题目**： 输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

  要求时间复杂度为O(n)。

- **思路**： 动态规划。设`F(i)`为数组中以`i`结尾的子数组的最大和，则 `F(i) = max(F(i - 1) + cur, cur)` .

```java
class Solution {
    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int result = nums[0];
        int curMax = nums[0];
        for (int i = 1; i < nums.length; i++) {
            curMax = Math.max(curMax + nums[i], nums[i]);
            result = Math.max(result, curMax);
        }
        return result;
    }
}
```



#### 把数组排成最小的数

- **题目**： 输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如， 输入： `[3, 30, 34, 5, 9]` ，输出 `"3033459"`.
- **思路**： heap。将所有数字转为字符串，然后加入小根堆中，再依次poll出来即可。所以问题的关键在于**比较器** 。如果两个字符串前面一些数字不等（比如：123 和 1278），那很好比较，但是存在第一个字符串是第二个字符串的子集的情况（比如：234 和  2342或2345），这个时候比较大小的时候就需要遍历完某个字符串之后再遍历一遍。（即比较 234 **234** 和 2342**2342**）

```java
class Solution {
    public String minNumber(int[] nums) {
        if (nums == null || nums.length == 0) {
            return null;
        }
        PriorityQueue<String> heap = new PriorityQueue<>(new Comparator<String>() {
            public int compare(String o1, String o2) {
                int len1 = o1.length();
                int len2 = o2.length();
                int i = 0;
                int j = 0;
                while (i != len1 && j != len2) {
                    if (o1.charAt(i) > o2.charAt(j)) {
                        return 1;
                    } else if (o1.charAt(i) < o2.charAt(j)) {
                        return -1;
                    }
                    i++;
                    j++;
                    if (i == len1 && j != len2) {
                        i = 0;
                    }
                    if (i != len1 && j == len2) {
                        j = 0;
                    }
                }
                return 0;
            }
        });
        for (int i : nums) {
            heap.add(String.valueOf(i));
        }
        StringBuilder sb = new StringBuilder();
        while (!heap.isEmpty()) {
            sb.append(heap.poll());
        }
        return sb.toString();
    }
}
```



#### 第一个只出现一个的字符

- **题目**： 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。
- **思路**： 略。

```java
class Solution {
    public char firstUniqChar(String s) {
        if (s == null || s.length() == 0) {
            return ' ';
        }
        Queue<Character> queue = new LinkedList<>();
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            int times = map.getOrDefault(ch, 0);
            if (times == 0) {
                queue.add(ch);
            }
            map.put(ch, times + 1);
        }
        while (!queue.isEmpty()) {
            char ch = queue.poll();
            if (map.get(ch) == 1) {
                return ch;
            }
        }
        return ' ';
    }
}
```



#### 数组中数字出现的次数

- **题目**： 一个整型数组 `nums` 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
- **思路**： 异或。将数组中所有的数字都进行异或操作，得到的结果就是两个只出现一次的数字的异或值。由于该两个数字不同，因此必定存在某个bit (一个是0一个是1) 使得异或值的某个bit值为1。只要将该bit不同的值分为两个部分，然后分别异或即可得到最终的结果。

```java
class Solution {
    public int[] singleNumbers(int[] nums) {
        int temp = 0;
        for (int i : nums) {
            temp ^= i;
        }
        int index = 1;
        while ((temp & index) == 0) {
            index <<= 1;
        }
        int num1 = 0;
        int num2 = 0;
        for (int i : nums) {
            if ((i & index) == 0) {
                num1 ^= i;
            } else {
                num2 ^= i;
            }
        }
        return new int[] {num1, num2};
    }
}
```



#### 在排序数组中查找数字

- **题目**： 统计一个数字在排序数组中出现的次数。
- **思路**： 二分查找。要统计某个数出现的次数，只要找到第一个和最后一个即可。查找第一个数稍微修改一下二分查找即可，最后一个可以通过查找target + 1 来实现。

```java
class Solution {
    public int search(int[] nums, int target) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int first = findFirst(nums, target);
        if (nums[first] != target) {
            return 0;
        }
        int last = findFirst(nums, target + 1);
        return nums[first] == nums[last] ? last - first + 1 : last - first;
    }

    private int findFirst(int[] nums, int target) {
        int low = 0; 
        int high = nums.length - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] < target) {
                low = mid + 1;
            } else if (nums[mid] > target) {
                high = mid - 1;
            } else {
                high = mid;
            }
        }
        return low;
    }
}
```



#### 和为S的两个数字

- **题目**： 输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。
- **思路**： 二分。

```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        if (nums == null || nums.length < 2) {
            return null;
        }
        int low = 0;
        int high = nums.length - 1;
        while (low < high) {
            if (nums[low] + nums[high] == target) {
                return new int[] {nums[low], nums[high]};
            } else if (nums[low] + nums[high] > target) {
                high--;
            } else {
                low++;
            }
        }
        return null;
    }
}
```



#### 滑动窗口的最大值

- **题目**： 给定一个数组 `nums` 和滑动窗口的大小 `k`，请找出所有滑动窗口里的最大值。
- **思路**： 双向队列。用一个单调的双向队列来保存滑动窗口中可能的最大值。双向队列的头元素保存当前窗口的最大值。对于下一个加进来的元素，将其与队列中的元素从尾到头比较一遍，将小于当前元素的值全部删除，只保留可能成为最大值的元素。

```java
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if (nums == null || nums.length == 0 || nums.length < k) {
            return new int[] {};
        }
        int[] result = new int[nums.length - k + 1];
        Deque<Integer> deque = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            while (!deque.isEmpty() && nums[deque.getLast()] <= nums[i]) {
                deque.removeLast();
            }
            if (!deque.isEmpty() && deque.getFirst() < i - k + 1) {
                deque.removeFirst();
            }
            deque.add(i);
            if (i >= k - 1) {
                result[i - k + 1] = nums[deque.getFirst()];
            }
        }
        return result;
    }
}
```



#### 两个链表的第一个公共节点

- **题目**： 输入两个链表，找出它们的第一个公共节点。
- **思路**： A链表走完自己的路再走B，B链表走完自己的路再走A的路。如果有公共节点，那么肯定会在第一个公共节点相遇，因为两个链表所走的路程是一样的。如果没有公共节点，那么都会以null结束。

```java
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode curA = headA;
        ListNode curB = headB;
        while (curA != curB) {
            curA = curA == null ? headB : curA.next;
            curB = curB == null ? headA : curB.next;
        }
        return curA;
    }
}
```



#### 二叉树的深度

- **题目**： 输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。
- **思路**： 二叉树的深度等于根节点加上最大子树的高度。

```java
class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }
}
```



#### 0 ~ n - 1中缺失的数字

- **题目**： 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。
- **思路**： 二分。如果`nums[i] = i ` ，那么就说明 i 前面的所有数都是存在的，那么就在后面去找，直到找到最左边那个 `nums[i] != i`的数。

```java
class Solution {
    public int missingNumber(int[] nums) {
        if(nums == null || nums.length == 0) {
            return -1;
        }
        int low = 0;
        int high = nums.length - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (nums[mid] == mid) {
                low = mid + 1;
            } else if (nums[mid] > mid) {
                high = mid;
            } else {
                return -1;
            }
        }
        return low == nums[low] ? nums.length : low;
    }
}
```



#### 二叉搜索树的第K大节点

- **题目**： 给定一棵二叉搜索树，请找出其中第k大的节点。
- **思路**： 最简单的就是中序遍历一遍，将结果放在一个数组中，然后返回倒数第K个。更好的方法是参考中序遍历的做法，依照  **右子树 **->  **根节点**  -> **左子树** 的顺序遍历一遍，到达第K个返回即可。

```java
class Solution {
    private int result = -1;
    private int count = 0;

    public int kthLargest(TreeNode root, int k) {
        inorder(root, k);
        return result;
    }

    private void inorder(TreeNode root, int k) {
        if (root == null) {
            return ;
        }
        inorder(root.right, k);
        if (++count == k) {
            result = root.val;
            return ;
        }
        inorder(root.left, k);
    }
}
```



#### 数组中的逆序对

- **题目**： 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。例如：输入：`[7, 5, 6, 4]`，输出5.
- **思路**： 最简单的方法就是两层for循环暴力解决，但是时间复杂度为`O(n^2)`，会超时。如果数组可以分成两半有序的数组，那么求这两个数组之间的逆序对个数将会简单很多。因此可以利用归并排序的思想，将数组的两半分别排序并计算子数组内逆序对的个数，那么总逆序对个数就等于两个子数组内部的逆序对个数加上它们之间的逆序对个数。为了方便计算，排序为倒序。

```java
class Solution {
    private int result = 0;
    private int[] arr;

    public int reversePairs(int[] nums) {
        if (nums == null || nums.length < 2) {
            return result;
        }
        arr = new int[nums.length];
        reversePairs(nums, 0, nums.length - 1);
        return result;
    }

    private void reversePairs(int[] nums, int low, int high) {
        if (low >= high) {
            return ;
        }
        int mid = low + (high - low) / 2;
        reversePairs(nums, low, mid);
        reversePairs(nums, mid + 1, high);
        merge(nums, low, mid, high);
    }

    private void merge(int[] nums, int low, int mid, int high) {
        int start1 = low, end1 = mid;
        int start2 = mid + 1, end2 = high;
        int cur = low;
        while (start1 <= end1 && start2 <= end2) {
            if (nums[start1] > nums[start2]) {
                result += end2 - start2 + 1;
                arr[cur] = nums[start1++];
            } else {
                arr[cur] = nums[start2++];
            }
            cur++;
        }
        while (start1 <= end1) {
            arr[cur++] = nums[start1++];
        }
        while (start2 <= end2) {
            arr[cur++] = nums[start2++];
        }
        for (int i = low; i <= high; i++) {
            nums[i] = arr[i];
        }
    }
}
```



#### 扑克牌中的顺子

- **题目**：从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。例如：输入`[1, 2, 3, 4, 5]`或者`[0, 0, 1, 2, 5]`，则返回true。
- **思路**：
  1. 将数组排序，然后统计0的个数，然后再判断所有非0的元素两两之间的差的总和是否大于0的个数，如果大于，则false，否则true。
  2. 如果又重复，直接返回false。然后判断非0的元素中最大值和最小值之间的差小于5即可。（0直接略过）。

```java
class Solution {
    public boolean isStraight(int[] nums) {
        if (nums == null || nums.length != 5) {
            return false;
        }
        Arrays.sort(nums);
        int index = 0;
        int numOfZero = 0;
        while (index < nums.length - 1) {
            if (nums[index] == 0) {
                numOfZero++;
                index++;
                continue;
            }
            if (nums[index + 1] == nums[index]) {
                return false;
            }
            int diff = nums[index + 1] - nums[index] - 1;
            if (diff > numOfZero) {
                return false;
            }
            numOfZero -= diff;
            index++;
        }
        return true;
    }
}
```



```java
class Solution {
    public boolean isStraight(int[] nums) {
        if (nums == null || nums.length != 5) {
            return false;
        }
        Set<Integer> set = new HashSet<>();
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        for (int i : nums) {
            if (i == 0) {
                continue;
            }
            if (set.contains(i)) {
                return false;
            }
            set.add(i);
            max = Math.max(i, max);
            min = Math.min(i, min);
        }
        return max - min < 5;
    }
}
```

