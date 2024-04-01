TOC
- [数组](#数组)
- [链表](#链表)
- [哈希表](#哈希表)
- [字符串](#字符串)
- [栈和队列](#栈和队列)
- [二叉树](#二叉树)
- [回溯算法](#回溯算法)
- [贪心算法](#贪心算法)
- [动态规划](#动态规划)
- [单调栈](#单调栈)
- [图论](#图论)


### 数组
**704. 二分查找**
```py
def search(self, nums: List[int], target: int) -> int:
    i,j = 0,len(nums)-1
    while i<=j:
        mid = (i+j)//2
        if target==nums[mid]: return mid
        elif target<nums[mid]: j = mid-1
        else: i=mid+1
    return -1
```
**35. 搜索插入位置**
```py
def searchInsert(self, nums: List[int], target: int) -> int:
    i,j=0,len(nums)-1
    while(i<=j):
        mid = i+(j-i)//2
        if target==nums[mid]: return mid
        elif target<nums[mid]: j=mid-1
        else: i=mid+1
    return i
```

**34. 在排序数组中查找元素的第一个和最后一个位置**
```py
def searchRange(self, nums: List[int], target: int) -> List[int]:
    i,j = 0,len(nums)-1
    while(i<=j):
        mid = i+(j-i)//2
        if target<=nums[mid]:j=mid-1
        else: i=mid+1
    left = i
    i,j = 0,len(nums)-1
    while(i<=j):
        mid = i+(j-i)//2
        if target>=nums[mid]: i=mid+1
        else: j = mid-1
    right = j
    if left==len(nums) or right==-1: return [-1,-1]
    # one side is ok!
    if nums[left]!=target: return [-1,-1]
    return [left,right]
```
**69. x 的平方根**
```py
def mySqrt(self, x: int) -> int:
    if x==0 or x==1: return x
    i,j = 1,x
    while(i<=j):
        mid = i+(j-i)//2
        # j position
        if x==mid*mid: return mid
        elif x<mid*mid: j=mid-1
        else: i=mid+1
    return j
```
**367. 有效的完全平方数**
```py
def isPerfectSquare(self, num: int) -> bool:
    i,j = 1,num
    while i<=j:
        mid = (i+j)//2
        temp = mid*mid
        if temp==num: return True
        elif temp<num: i=mid+1
        else: j=mid-1
    return False
```
**26. 删除有序数组中的重复项**
```py
def removeDuplicates(self, nums: List[int]) -> int:
    # i指向下一个需要被填充的位置
    i=1
    for j in range(1,len(nums)):
        if nums[j]==nums[i-1]:continue
        nums[i]=nums[j]
        i+=1
    return i
```
**27. 移除元素**
```py
def removeElement(self, nums: List[int], val: int) -> int:
    i,j = 0,len(nums)-1
    while(i<=j):
        while(i<=j and nums[i]!=val):i+=1
        while(i<=j and nums[j]==val):j-=1
        if(i<=j):
            nums[i]=nums[j]
            i+=1
            j-=1
    return i
```
**283. 移动零**
```py
def moveZeroes(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    i = 0
    for j in range(len(nums)):
        if nums[j]!=0:
            if i!=j: nums[i] = nums[j]
            i+=1
    nums[i:] = [0]*(len(nums)-i)
```
**844. 比较含退格的字符串**
```py
def backspaceCompare(self, s: str, t: str) -> bool:
    n1 = len(s)
    n2 = len(t)

    i = n1-1
    j = n2-1

    skipA = 0
    skipB = 0

    while i>=0 or j>=0:
        while (i>=0) and (s[i]=='#' or skipA):
            if s[i]=='#': skipA+=1
            else: skipA-=1
            i-=1

        while (j>=0) and (t[j]=='#' or skipB):
            if t[j]=='#': skipB+=1
            else: skipB-=1
            j-=1
            
        if i<0 and j<0: return True
        if i<0 or j<0: return False
        if s[i] != t[j]: return False
        i-=1
        j-=1

    return True
```
**977. 有序数组的平方**
```py
def sortedSquares(self, nums: List[int]) -> List[int]:
    n = len(nums)
    i,j = 0,n-1
    res = [0]*n
    a,b = nums[i]**2,nums[j]**2
    for k in range(n-1,-1,-1):
        if a>=b:
            res[k] = a
            i+=1
            if i<=j: a=nums[i]**2
        else:
            res[k]=b
            j-=1
            if j>=i: b=nums[j]**2
    return res
```
**209. 长度最小的子数组**
```py
def minSubArrayLen(self, target: int, nums: List[int]) -> int:
    n = len(nums)
    ans = n+1
    i,j = 0,0
    sums = 0
    while j<n:
        sums+=nums[j]
        while sums>=target:
            ans = min(ans,j-i+1)
            sums-=nums[i]
            i+=1
        j+=1
    return ans if ans<n+1 else 0
```
**904. 水果成篮**
```py
def totalFruit(self, fruits: List[int]) -> int:
    buckets = Counter()
    ans = 0
    i,j = 0,0
    n = len(fruits)
    while j<n:
        buckets[fruits[j]]+=1
        while len(buckets)>2:
            buckets[fruits[i]]-=1
            if not buckets[fruits[i]]:
                buckets.pop(fruits[i])
            i+=1
        j+=1
        ans = max(ans,j-i)
    return ans
```
**76. 最小覆盖子串**
```py
def minWindow(self, s: str, t: str) -> str:
    n = len(s)
    ans = ""
    ans_length = n+1
    tar_num = len(t)
    tar_dict = defaultdict(int)
    for c in t: tar_dict[c]+=1
    i,j=0,0
    while j<n:
        tar_dict[s[j]]-=1
        if tar_dict[s[j]]>=0: tar_num-=1
        while tar_num==0:
            if j-i+1<ans_length:
                ans=s[i:j+1]
                ans_length = j-i+1
            tar_dict[s[i]]+=1
            if tar_dict[s[i]]>0: tar_num+=1
            i+=1
        j+=1
    return ans    
```
**59. 螺旋矩阵 II**
```py
def generateMatrix(self, n: int) -> List[List[int]]:
    ans = [[0]*n for _ in range(n)]
    k = 1
    l,r,t,b = 0,n-1,0,n-1
    while k<=n*n:
        for j in range(l,r+1):
            ans[t][j]=k
            k+=1
        t+=1
        for i in range(t,b+1):
            ans[i][r]=k
            k+=1
        r-=1
        for j in range(r,l-1,-1):
            ans[b][j]=k
            k+=1
        b-=1
        for i in range(b,t-1,-1):
            ans[i][l]=k
            k+=1
        l+=1
    return ans        
```
**54. 螺旋矩阵**
```py
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    m,n = len(matrix),len(matrix[0])
    ans = []

    l,r = 0,n-1
    t,b = 0,m-1

    while True:
        if l>r: break
        for j in range(l,r+1): ans.append(matrix[t][j])
        t+=1
        if t>b: break
        for i in range(t,b+1): ans.append(matrix[i][r])
        r-=1
        if r<l: break
        for j in range(r,l-1,-1): ans.append(matrix[b][j])
        b-=1
        if b<t: break
        for i in range(b,t-1,-1): ans.append(matrix[i][l])
        l+=1
    return ans     
```
### 链表
**203. 移除链表元素**
```py
def removeElements(self, head: Optional[ListNode], val: int):
    dummy = ListNode()
    dummy.next = head
    cur = dummy
    while cur and cur.next:
        if cur.next.val == val: cur.next = cur.next.next
        else: cur = cur.next
    return dummy.next
```
**707. 设计链表**
```py
class ListNode:
    def __init__(self,val=0,next=None):
        self.val = val
        self.next = next

class MyLinkedList:

    def __init__(self):
        self.dummy = ListNode()
        self.size=0

    def get(self, index: int) -> int:
        if index<0 or index>=self.size: return -1
        cur = self.dummy.next
        for _ in range(index):
            cur = cur.next
        return cur.val

    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0,val)

    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self.size,val)

    def addAtIndex(self, index: int, val: int) -> None:
        if index>self.size: return 
        pre = self.dummy
        for _ in range(index):
            pre = pre.next
        pre.next = ListNode(val,pre.next)
        self.size+=1

    def deleteAtIndex(self, index: int) -> None:
        if index<0 or index>=self.size: return 
        pre = self.dummy
        for _ in range(index):
            pre = pre.next
        pre.next = pre.next.next
        self.size-=1
```
**206. 反转链表**
```py
def reverseList(self, head: Optional[ListNode]):
    dummy = ListNode()
    cur = head
    while cur:
        save = cur.next
        cur.next = dummy.next
        dummy.next = cur
        cur = save
    return dummy.next
```
**24. 两两交换链表中的节点**
```py
def swapPairs(self, head: Optional[ListNode]):
    if not head: return None
    if not head.next: return head

    dummy = ListNode()
    dummy.next = head
    cur = dummy
    while cur.next and cur.next.next:
        a,b = cur.next,cur.next.next
        cur.next = b
        a.next = b.next
        b.next = a
        cur = a
    return dummy.next
```
**19. 删除链表的倒数第 N 个结点**
```py
def removeNthFromEnd(self, head: Optional[ListNode], n: int):
    dummy = ListNode()
    dummy.next = head
    slow = dummy
    fast = head
    for _ in range(n): fast = fast.next
    while fast:
        slow = slow.next
        fast = fast.next
    slow.next = slow.next.next
    return dummy.next 
```
**160. 链表相交**
```py
def getIntersectionNode(self, headA: ListNode, headB: ListNode):
    A,B = headA,headB
    while A!=B:
        A = A.next if A else headB
        B = B.next if B else headA
    return A 
```
**142. 环形链表 II**
```py
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]):
        if not head: return None
        slow,fast = head,head
        while True:
            slow = slow.next
            if fast and fast.next: 
                fast = fast.next.next
            else: return None
            if slow==fast: break
        slow = head
        while slow!=fast:
            slow=slow.next
            fast=fast.next
        return slow
```
### 哈希表
**242. 有效的字母异位词**
```py
def isAnagram(self, s: str, t: str) -> bool:
    return Counter(s)==Counter(t)
```
```py
def isAnagram(self, s: str, t: str) -> bool:
    cache = [0]*26
    for c in s: cache[ord(c)-ord('a')]+=1
    for c in t: cache[ord(c)-ord('a')]-=1

    for n in cache:
        if n: return False
    return True
```
**1002. 查找共用字符**
```py
def commonChars(self, words: List[str]) -> List[str]:
    cache = [0]*26
    for c in words[0]: cache[ord(c)-ord('a')]+=1
    temp = [0]*26
    for w in words[1:]:
        for i in range(26): temp[i]=0
        for c in w: temp[ord(c)-ord('a')]+=1
        for i in range(26):
            if cache[i]==0: continue
            elif cache[i]>temp[i]:
                cache[i]=temp[i]
    res = []
    for i,n in enumerate(cache):
        for _ in range(n):
            res.append(chr(i+ord('a')))
    return res
```
```py
def commonChars(self, words: List[str]) -> List[str]:
    delete = []
    res = Counter(words[0])

    for w in words[1:]:
        temp = Counter(w)
        for key in res.keys():
            if key not in temp:
                delete.append(key)
            elif temp[key]<res[key]:
                res[key] = temp[key]
        for key in delete:
            del res[key]
    return list(res.elements())
```
**349. 两个数组的交集**
```py
def intersection(self, nums1: List[int], nums2: List[int]):
    return list(set(nums1)&set(nums2))
```
```py
def intersection(self, nums1: List[int], nums2: List[int]):
    set1 = set(nums1)
    ans = set()
    for n in nums2:
        if n in set1: 
            ans.add(n)
    return list(ans)
```
**202. 快乐数**
```py
def isHappy(self, n: int) -> bool:
    if n==1: return True
    seen = set()
    while True:
        temp = 0
        while n:
            temp+=(n%10)**2
            n//=10
        if temp==1: return True
        elif temp in seen: return False
        seen.add(temp)
        n = temp
        
```
**1. 两数之和**
```py
def twoSum(self, nums: List[int], target: int):
    partner = dict()
    for i,n in enumerate(nums):
        if (target-n) in partner: return [i,partner[target-n]]
        partner[n]=i
    return []
```
**454. 四数相加 II**
```py
def fourSumCount(self, nums1, nums2, nums3, nums4):
    cache = Counter((i+j) for i in nums1 for j in nums2)
    res = 0
    for k in nums3:
        for l in nums4:
            if (-k-l) in cache:
                res+=cache[-k-l]
    return res
```
**383. 赎金信**
```py
def canConstruct(self, ransomNote: str, magazine: str) -> bool:
    return Counter(ransomNote)<=Counter(magazine)
```
**15. 三数之和**
```py
def threeSum(self, nums: List[int]) -> List[List[int]]:
    ans = []
    nums.sort()
    n = len(nums)
    i = 0
    while i<n-2 and nums[i]<=0:
        k = i+1
        j = n-1
        while k<j:
            temp = nums[i]+nums[k]+nums[j]
            if temp == 0: 
                ans.append([nums[i],nums[k],nums[j]])
                k+=1
                while k<j and nums[k-1]==nums[k]:k+=1
            elif temp>0:j-=1
            else: k+=1
        i+=1
        while i<n-2 and nums[i-1]==nums[i]:i+=1
    return ans
```
**18. 四数之和**
```py
def fourSum(self, nums: List[int], target: int):
    n = len(nums)
    if n<4: return []
    nums.sort()
    res = []

    for i in range(n-3):
        if i>0 and nums[i]==nums[i-1]: continue
        # if nums[i]>target: break 有负数
        if nums[i]+nums[i+1]+nums[i+2]+nums[i+3]>target: break
        if nums[i]+nums[n-1]+nums[n-2]+nums[n-3]<target: continue
        for j in range(i+1,n-2):
            if j>(i+1) and nums[j]==nums[j-1]: continue
            # if nums[i]+nums[j]>target: break
            if nums[i]+nums[j]+nums[j+1]+nums[j+2]>target: break
            if nums[i]+nums[j]+nums[n-1]+nums[n-2]<target: continue
            k,l = j+1,n-1
            while(k<l):
                if nums[i]+nums[j]+nums[k]+nums[l]<target:
                    k+=1
                elif nums[i]+nums[j]+nums[k]+nums[l]>target:
                    l-=1
                else:
                    res.append([nums[i],nums[j],nums[k],nums[l]])
                    k+=1
                    l-=1
                    while k<l and nums[k]==nums[k-1]:k+=1
                    while k<l and nums[l]==nums[l+1]:l-=1
    return res
```
### 字符串
**344. 反转字符串**
```py
def reverseString(self, s: List[str]) -> None:
    """
    Do not return anything, modify s in-place instead.
    """
    i,j = 0,len(s)-1
    while(i<j):
        s[i],s[j]=s[j],s[i]
        i+=1
        j-=1
```
**541. 反转字符串 II**
```py
def reverseStr(self, s: str, k: int) -> str:
    res = list(s)
    n = len(s)
    for i in range(0,n,2*k):
        l,r = i,min(i+k,n)-1
        while(l<r):
            res[l],res[r]=res[r],res[l]
            l+=1
            r-=1
    return ''.join(res)
```
**151. 反转字符串中的单词**
```py
class Solution:
    def reverseWords(self, s: str) -> str:
        # split(ch,max_split_num)

        return ' '.join(s.strip().split()[::-1])
```
```py
def reverseWords(self, s: str) -> str:
    res = []
    n = len(s)
    j,i = n-1,n
    while j>=0:
        while j>=0 and s[j]==' ':j-=1
        if j>=0: i=j
        while j>=0 and s[j]!=' ':j-=1
        if i<n:
            res.append(s[j+1:i+1])
            i=n
    return ' '.join(res)
```
**28 找出字符串中第一个匹配项的下标**
```py
def strStr(self, haystack: str, needle: str) -> int:
    return haystack.find(needle)
```
**459. 重复的子字符串**
```py
def repeatedSubstringPattern(self, s: str) -> bool:
    # find(val,start,end)
    return ((s+s).find(s,1)!=len(s))
```
### 栈和队列
**232. 用栈实现队列**
```py
class MyQueue:

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x: int) -> None:
        self.stack1.append(x)

    def pop(self) -> int:
        if self.stack2: return self.stack2.pop()
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        return self.stack2.pop()

    def peek(self) -> int:
        if self.stack2: return self.stack2[-1]
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        return self.stack2[-1]

    def empty(self) -> bool:
        return (not self.stack1 and not self.stack2)


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```
**225. 用队列实现栈**
```py
class MyStack:

    def __init__(self):
        self.queue = deque([])

    def push(self, x: int) -> None:
        if not self.queue:
            self.queue.append(x)
            return 
        n = len(self.queue)
        self.queue.append(x)
        for _ in range(n):
            self.queue.append(self.queue.popleft())

    def pop(self) -> int:
        return self.queue.popleft()

    def top(self) -> int:
        return self.queue[0]

    def empty(self) -> bool:
        return not self.queue


# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()
```
**20. 有效的括号**
```py
def isValid(self, s: str) -> bool:
    stack = ['?']
    dic = {'(':')','{':'}','[':']','?':'?'}
    for c in s:
        if c in dic:
            stack.append(c)
        else:
            if dic[stack[-1]]!=c: return False
            stack.pop()
    return len(stack)==1
```
**1047. 删除字符串中的所有相邻重复项**
```py
def removeDuplicates(self, s: str) -> str:
    stack = [s[0]]
    for c in s[1:]:
        if stack and stack[-1]==c:
            stack.pop()
        else:
            stack.append(c)
    return ''.join(stack)
```
**150. 逆波兰表达式求值**
```py
def evalRPN(self, tokens: List[str]) -> int:
    dic = {
        '+':add,
        '-':sub,
        '*':mul,
        '/':lambda x,y: int(x/y), #python趋向-无穷
    }
    stack = []
    for c in tokens:
        try:
            num = int(c)
        except ValueError:
            n2 = int(stack.pop())
            n1 = int(stack.pop())
            num = dic[c](n1,n2)
        finally:
            stack.append(num)
    return stack[0]

```
**239. 滑动窗口最大值**
```py
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    dq = deque([])
    for i in range(k):
        while dq and nums[dq[-1]]<=nums[i]:
            dq.pop()
        dq.append(i)
    res = [nums[dq[0]]]
    n = len(nums)
    for i in range(k,n):
        while dq and nums[dq[-1]]<=nums[i]:
            dq.pop()
        dq.append(i)
        if dq[0]<=i-k:dq.popleft()
        res.append(nums[dq[0]])
    return res
```
**347. 前 K 个高频元素**
```py
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    count = Counter(nums)
    heap = []
    for key,val in count.items():
        if len(heap)==k:
            if val>heap[0][0]:
                heapq.heapreplace(heap,(val,key))
        else:
            heapq.heappush(heap,(val,key))
    return [it[1] for it in heap]
```
### 二叉树
**144. 二叉树的前序遍历**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def preorderTraversal(self, root: Optional[TreeNode]):

    def dfs(node):
        if not node: return 
        res.append(node.val)
        dfs(node.left)
        dfs(node.right)
    
    res = []
    dfs(root)
    return res
```
```py
def preorderTraversal(self, root) -> List[int]:
    if not root: return []
    stack = [root]
    ans = []
    while stack:
        node = stack.pop()
        ans.append(node.val)
        if node.right: stack.append(node.right)
        if node.left: stack.append(node.left)
    return ans
```
**145. 二叉树的后序遍历**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def postorderTraversal(self, root: Optional[TreeNode]):
    def dfs(node):
        if not node: return 
        dfs(node.left)
        dfs(node.right)
        res.append(node.val)
    
    res = []
    dfs(root)
    return res
```
```py
def postorderTraversal(self, root) -> List[int]:
        if not root:
            return list()
        
        res = list()
        stack = list()
        prev = None

        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if not root.right or root.right == prev:
                res.append(root.val)
                prev = root
                root = None
            else:
                stack.append(root)
                root = root.right
        
        return res
```
**94. 二叉树的中序遍历**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def inorderTraversal(self, root: Optional[TreeNode]):
    def dfs(node):
        if not node: return 
        dfs(node.left)
        res.append(node.val)
        dfs(node.right)
        
    res = []
    dfs(root)
    return res
```
```py
def inorderTraversal(self, root) -> List[int]:
    stack = []
    ans = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        ans.append(root.val)
        root = root.right
    return ans
```
**102. 二叉树的层序遍历**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def levelOrder(self, root: Optional[TreeNode]):
    if not root: return []
    dq = deque([root])
    res = []
    while dq:
        temp = list()
        for _ in range(len(dq)):
            node = dq.popleft()
            temp.append(node.val)
            if node.left: dq.append(node.left)
            if node.right: dq.append(node.right)
        res.append(temp)
    return res
```
**107. 二叉树的层序遍历 II**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def levelOrderBottom(self, root: Optional[TreeNode]):
    if not root: return []
    dq = deque([root])
    res = []
    while dq:
        temp = list()
        for _ in range(len(dq)):
            node = dq.popleft()
            temp.append(node.val)
            if node.left: dq.append(node.left)
            if node.right: dq.append(node.right)
        res.append(temp)
    return res[::-1]
```
**199. 二叉树的右视图**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    def dfs(node,level):
        if not node: return 
        if level==len(res): res.append(node.val)
        dfs(node.right,level+1)
        dfs(node.left,level+1)
    res = []
    dfs(root,0)
    return res
```
**637. 二叉树的层平均值**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def averageOfLevels(self, root: Optional[TreeNode]):
    res = []
    dq = deque([root])

    while dq:
        sums = 0
        num = len(dq)
        for _ in range(num):
            node = dq.popleft()
            sums+=node.val
            if node.left: dq.append(node.left)
            if node.right: dq.append(node.right)
        res.append(sums/num)
    return res
```
**429. N 叉树的层序遍历**
```py
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

def levelOrder(self, root: 'Node') -> List[List[int]]:
    if not root: return []
    dq = deque([root])
    res = []
    
    while dq:
        temp = list()
        for _ in range(len(dq)):
            node = dq.popleft()
            temp.append(node.val)
            for child in node.children:
                dq.append(child)
        res.append(temp)
    return res
        
```
**515. 在每个树行中找最大值**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def largestValues(self, root: Optional[TreeNode]):
    if not root: return []
    dq = deque([root])
    res = []

    while dq:
        maxVal = -inf
        for _ in range(len(dq)):
            node = dq.popleft()
            if node.val>maxVal: maxVal = node.val
            if node.left: dq.append(node.left)
            if node.right: dq.append(node.right)
        res.append(maxVal)
    return res
```
```py
def largestValues(self, root: Optional[TreeNode]) -> List[int]:
    ans = []
    def dfs(node,level):
        if not node: return 
        n = len(ans)
        if level==n: ans.append(node.val)
        elif node.val>ans[level]:
            ans[level]=node.val
        dfs(node.left,level+1)
        dfs(node.right,level+1)
    dfs(root,0)
    return ans
```
**116. 填充每个节点的下一个右侧节点指针**
```py
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

def connect(self, root: 'Optional[Node]'):
    if not root: return 
    leftmost = root
    while leftmost.left:
        cur  = leftmost
        while cur:
            cur.left.next = cur.right
            if cur.next:
                cur.right.next = cur.next.left
            cur = cur.next
        leftmost = leftmost.left
    return root
        
        
```
**117. 填充每个节点的下一个右侧节点指针 II**
```py
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

def connect(self, root: 'Node') -> 'Node':
    if not root: return None

    node = root
    while node:
        last = None
        nextStart = None

        p = node
        while p:
            if p.left:
                if last: last.next = p.left
                if not nextStart: nextStart = p.left
                last = p.left
            if p.right:
                if last: last.next = p.right
                if not nextStart: nextStart = p.right
                last = p.right
            p = p.next
        node = nextStart
    return root
        
```
**104. 二叉树的最大深度**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def maxDepth(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    return 1+max(self.maxDepth(root.left),self.maxDepth(root.right))
        
```
**111. 二叉树的最小深度**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def minDepth(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    dq = deque([root])
    ans = 0
    
    while dq:
        ans+=1
        for _ in range(len(dq)):
            node = dq.popleft()
            if not node.left and not node.right: return ans
            if node.left: dq.append(node.left)
            if node.right: dq.append(node.right)      
```
**226. 翻转二叉树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def invertTree(self, root: Optional[TreeNode]):
    if not root: return 
    self.invertTree(root.left)
    self.invertTree(root.right)
    root.left,root.right = root.right,root.left
    return root
```
**101. 对称二叉树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    def dfs(l,r):
        if not l and not r: return True
        if not l or not r: return False
        if l.val!=r.val: return False
        return dfs(l.left,r.right) and dfs(l.right,r.left)
    return dfs(root.left,root.right)
```
**100. 相同的树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def isSameTree(self, p, q) -> bool:
    def dfs(l,r):
        if not l and not r: return True
        if not l or not r: return False
        if l.val!=r.val: return False
        return dfs(l.left,r.left) and dfs(l.right,r.right)
    return dfs(p,q)
```
**572. 另一棵树的子树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def dfs(self,l,r):
        if not l and not r: return True
        if not l or not r: return False
        if l.val!=r.val: return False
        return self.dfs(l.left,r.left) and self.dfs(l.right,r.right)

def isSubtree(self, root, subRoot) -> bool:
    if not root: return False
    return self.dfs(root,subRoot) or self.isSubtree(root.left,subRoot) \
        or self.isSubtree(root.right,subRoot)    
```
```py
def isSame(self,l,r):
    if not l and not r: return True
    if not l or not r: return False
    if l.val!=r.val: return False
    return self.isSame(l.left,r.left) and self.isSame(l.right,r.right)

def isSubtree(self, root, subRoot) -> bool:
    if not root: return False
    if root.val==subRoot.val and self.isSame(root,subRoot):
        return True
    else: return self.isSubtree(root.left,subRoot) \
            or self.isSubtree(root.right,subRoot)
```

**559. N 叉树的最大深度**
```py
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

def maxDepth(self, root: 'Node') -> int:
    if not root: return 0

    dq = deque([root])
    ans = 0
    while dq:
        for _ in range(len(dq)):
            node = dq.popleft()
            for child in node.children:
                dq.append(child)
        ans+=1
    return ans 
```
**222. 完全二叉树的节点个数**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def height(self,root):
    h = 0
    while root:
        root =  root.left
        h+=1
    return h

def countNodes(self, root: Optional[TreeNode]) -> int:
    if not root: return 0

    leftDepth = self.height(root.left)
    rightDepth = self.height(root.right)

    if leftDepth==rightDepth:
        return 1+(2**leftDepth-1)+self.countNodes(root.right)
    else:
        return 1+self.countNodes(root.left)+(2**rightDepth-1)
```
```py
def height(self,node):
    if not node: return 0
    h = 1
    while node.left:
        h+=1
        node = node.left
    return h

def countNodes(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    h1 = self.height(root.left)
    h2 = self.height(root.right)
    if h1==h2: return 2**h1+self.countNodes(root.right)
    else: return 2**h2+self.countNodes(root.left)
```
**110. 平衡二叉树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def isBalanced(self, root: Optional[TreeNode]) -> bool:

    def dfs(node):
        if not node: return 0
        leftHeight = dfs(node.left)
        rightHeight = dfs(node.right)

        if leftHeight==-1 or rightHeight==-1: return -1
        if abs(leftHeight-rightHeight)>1: return -1
        return 1+max(leftHeight,rightHeight)
    
    return dfs(root)!=-1
```
**257. 二叉树的所有路径**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def binaryTreePaths(self, root: Optional[TreeNode]):

    def dfs(node,temp):
        if not node.left and not node.right:
            ans.append(temp)
        if node.left: dfs(node.left,temp+'->'+str(node.left.val))
        if node.right: dfs(node.right,temp+'->'+str(node.right.val))
    
    ans = []
    dfs(root,str(root.val))
    return ans
```
**404. 左叶子之和**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
    def isLeaf(node):
        if not node.left and not node.right: return True
        else: return False

    def dfs(node):
        if node.left:
            if isLeaf(node.left):
                nonlocal sums
                sums+=node.left.val
            else: dfs(node.left)
        if node.right:
            dfs(node.right)
    
    sums = 0
    dfs(root)
    return sums
```
**513. 找树左下角的值**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def findBottomLeftValue(self, root) -> int:

    def dfs(node,level):
        nonlocal tar,ans
        if level == tar:     
            tar+=1
            ans = node.val
        if node.left: dfs(node.left,level+1)
        if node.right: dfs(node.right,level+1)
    
    tar = 0
    ans = root.val
    dfs(root,0)
    return ans
```
**112. 路径总和**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def hasPathSum(self, root, targetSum: int) -> bool:
    if not root: return False

    def dfs(node,tar):
        if not node.left and not node.right:
            return tar==node.val
        
        if node.left: 
            if dfs(node.left,tar-node.val): return True
        if node.right:
            if dfs(node.right,tar-node.val): return True
        return False
    return dfs(root,targetSum)
```
```py
def hasPathSum(self, root, targetSum: int) -> bool:
    if not root: return False
    if not root.left and not root.right:
        return root.val==targetSum
    if root.left and self.hasPathSum(root.left,targetSum-root.val):
        return True
    if root.right and self.hasPathSum(root.right,targetSum-root.val):
        return True
    return False
```
**113. 路径总和 II**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def pathSum(self, root, targetSum: int):
    if not root: return []

    def dfs(node,tar):
        if not node.left and not node.right:
            if tar==node.val:
                temp.append(node.val)
                ans.append(list(temp))
                temp.pop()
            return 
        temp.append(node.val)
        if node.left: dfs(node.left,tar-node.val)
        if node.right: dfs(node.right,tar-node.val)
        temp.pop()
    
    ans = []
    temp = []
    dfs(root,targetSum)
    return ans
```
**105. 从前序与中序遍历序列构造二叉树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def buildTree(self, preorder, inorder):
    inorderDict = {v:i for i,v in enumerate(inorder)}

    def dfs(m,l,r):
        if l>r: return None
        if l==r: return TreeNode(inorder[r])

        node = TreeNode(preorder[m])
        i = inorderDict[preorder[m]]
        node.left = dfs(m+1,l,i-1)
        node.right = dfs(m+i-l+1,i+1,r)
        return node

    return dfs(0,0,len(preorder)-1)
```
**106. 从中序与后序遍历序列构造二叉树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def buildTree(self, inorder, postorder):
    inorderDict = {v:i for i,v in enumerate(inorder)}

    def dfs(m,l,r):
        if l>r: return None
        if l==r: return TreeNode(inorder[r])

        node = TreeNode(postorder[m])
        i = inorderDict[postorder[m]]

        node.left = dfs(m-(r-i)-1,l,i-1)
        node.right = dfs(m-1,i+1,r)

        return node
    
    return dfs(len(postorder)-1,0,len(postorder)-1)
```
**654. 最大二叉树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def constructMaximumBinaryTree(self, nums: List[int]):
    stack = []

    for num in nums:
        node = TreeNode(num)
        while stack and stack[-1].val<num:
            node.left = stack.pop()
        if stack:
            stack[-1].right = node
        stack.append(node)
    return stack[0]
```
**617. 合并二叉树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def mergeTrees(self, root1, root2):
    if not root1: return root2
    if not root2: return root1

    node = TreeNode(root1.val+root2.val)
    node.left = self.mergeTrees(root1.left,root2.left)
    node.right = self.mergeTrees(root1.right,root2.right)
    return node
```
**700. 二叉搜索树中的搜索**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def searchBST(self, root, val: int):
    if not root: return None
    if root.val==val: return root
    elif val<root.val: return self.searchBST(root.left,val)
    else: return self.searchBST(root.right,val)
    
```
**98. 验证二叉搜索树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def isValidBST(self, root: Optional[TreeNode]) -> bool:
    pre = float('-inf')
    
    def dfs(node):
        if not node: return True
        if not dfs(node.left): return False
        nonlocal pre
        if node.val<=pre: return False
        pre = node.val
        return dfs(node.right)
    
    return dfs(root)
```
**530. 二叉搜索树的最小绝对差**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def getMinimumDifference(self, root) -> int:
    ans = float('inf')
    pre = -1

    def dfs(node):
        if not node: return
        dfs(node.left)
        nonlocal pre
        if pre==-1:
            pre = node.val
        else:
            nonlocal ans
            if abs(pre-node.val)<ans:
                ans = abs(pre-node.val)
        pre = node.val
        dfs(node.right)
    
    dfs(root)
    return ans
```
**501. 二叉搜索树中的众数**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def findMode(self, root) -> List[int]:
    res = []
    base = float('inf')
    count = 0
    maxCount = 0

    def dfs(node):
        if not node: return 
        dfs(node.left)

        nonlocal base,count,maxCount,res
        if node.val==base: count+=1
        else:
            base = node.val
            count=1
        
        if count==maxCount:
            res.append(base)
        
        if count>maxCount:
            maxCount = count
            res = [base]
        
        dfs(node.right)
    
    dfs(root)
    return res

```
**236. 二叉树的最近公共祖先**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

def lowestCommonAncestor(self, root, p, q):
    if not root: return None
    if root == p or root == q:
        return p if p==root else q
    l = self.lowestCommonAncestor(root.left,p,q)
    r = self.lowestCommonAncestor(root.right,p,q)

    if l and r: return root
    return l if l else r
        
```
**235. 二叉搜索树的最近公共祖先**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

def lowestCommonAncestor(self, root, p, q):
    if root==p or root==q: return root
    if root.val<p.val and root.val<q.val: return self.lowestCommonAncestor(root.right,p,q)
    if root.val>p.val and root.val>q.val: return self.lowestCommonAncestor(root.left,p,q)
    return root
        
```
**701. 二叉搜索树中的插入操作**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def insertIntoBST(self, root, val: int):
    if not root: return TreeNode(val)
    node = root 
    while node:
        if val<node.val:
            if not node.left:
                node.left = TreeNode(val)
                return root
            node = node.left
        else:
            if not node.right:
                node.right = TreeNode(val)
                return root
            node = node.right
    return root
                    
```
**450. 删除二叉搜索树中的节点**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def deleteNode(self, root, key: int):
    if not root: return None
    if key<root.val:
        root.left = self.deleteNode(root.left,key)
    elif key>root.val:
        root.right = self.deleteNode(root.right,key)
    elif not root.left or not root.right:
        return root.right if not root.left else root.left
    else:
        successor = root.right
        while successor.left:
            successor = successor.left

        # 这里顺序不能乱
        successor.right = self.deleteNode(root.right,successor.val)
        successor.left = root.left
        return successor
    return root
```
**669. 修剪二叉搜索树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def trimBST(self, root, low: int, high: int):
    while root and (root.val<low or root.val>high):
        root = root.right if root.val<low else root.left
    if not root: return None

    node = root
    while node.left:
        if node.left.val<low:
            node.left = node.left.right
        else:
            node = node.left
    
    node = root
    while node.right:
        if node.right.val>high:
            node.right = node.right.left
        else:
            node = node.right
    
    return root
```
**108. 将有序数组转换为二叉搜索树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def sortedArrayToBST(self, nums: List[int]):

    def dfs(l,r):
        if l>r: return None

        mid = (l+r)//2
        node = TreeNode(nums[mid])

        node.left = dfs(l,mid-1)
        node.right = dfs(mid+1,r)

        return node
    
    return dfs(0,len(nums)-1)
```
**538. 把二叉搜索树转换为累加树**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def convertBST(self, root):
    total = 0

    def inorder(node):
        if not node: return 

        inorder(node.right)
        nonlocal total
        tmp = node.val
        node.val+=total
        total+=tmp
        inorder(node.left)
    
    inorder(root)
    return root
```
### 回溯算法
**77. 组合**
```py
def combine(self, n: int, k: int) -> List[List[int]]:
    ans = []
    temp = []

    def dfs(x,count):
        if count==k:
            ans.append(list(temp))
            return 
        
        if n-x+1<k-count: return 

        for i in range(x,n+1):
            temp.append(i)
            dfs(i+1,count+1)
            temp.pop()
        
    dfs(1,0)
    return ans
```
```py
def combine(self, n: int, k: int) -> List[List[int]]:

    def dfs(x, num): 
        if num==k: 
            ans.append(list(temp))
            return
        for i in range(x,n):
            temp.append(i+1)
            dfs(i+1, num+1)
            temp.pop()
    
    ans = []
    temp = []
    dfs(0,0)
    return ans
```
**216. 组合总和 III**
```py
def combinationSum3(self, k: int, n: int) -> List[List[int]]:
    if k>n: return []
    ans = []
    temp = []

    def dfs(start,level,tar):
        if level==0:
            if not tar:
                ans.append(list(temp))
            return 
        
        for i in range(start,10):
            if i>tar: return
            else:
                temp.append(i)
                dfs(i+1,level-1,tar-i)
                temp.pop()
    dfs(1,k,n)
    return ans
```
**17. 电话号码的字母组合**
```py
def letterCombinations(self, digits: str) -> List[str]:
    dic = {
        '2':"abc",
        '3':"def",
        '4':"ghi",
        '5':"jkl",
        '6':"mno",
        '7':"pqrs",
        '8':"tuv",
        '9':"wxyz"
    }

    n = len(digits)
    if n==0: return []

    def dfs(i,tStr):
        if i==n:
            ans.append(tStr)
            return 
        
        for c in dic[digits[i]]:
            dfs(i+1,tStr+c)
    ans = []
    dfs(0,"")
    return ans       
```
**39. 组合总和**
```py
def combinationSum(self, candidates, target):
    candidates.sort()
    n = len(candidates)

    temp = []
    ans = []

    def dfs(start,tar):

        for i in range(start,n):
            if candidates[i]>tar: return 
            elif candidates[i]==tar:
                temp.append(candidates[i])
                ans.append(list(temp))
                temp.pop()
                return
            else:
                temp.append(candidates[i])
                dfs(i,tar-candidates[i])
                temp.pop()
    
    dfs(0,target)
    return ans
```
**40. 组合总和 II**
```py
def combinationSum2(self, candidates, target):

    candidates.sort()
    n = len(candidates)

    temp = []
    ans = []

    def dfs(start,tar):
        
        for i in range(start,n):
            if i>start and candidates[i]==candidates[i-1]:continue
            if candidates[i]>tar: return
            elif candidates[i]==tar:
                temp.append(candidates[i])
                ans.append(list(temp))
                temp.pop()
                return
            else:
                temp.append(candidates[i])
                dfs(i+1,tar-candidates[i])
                temp.pop()
    dfs(0,target)
    return ans
```
**131. 分割回文串**
```py
def partition(self, s: str) -> List[List[str]]:
    
    @cache
    def isPalindrome(i,j):
        while i<j:
            if s[i]!=s[j]: return False
            i+=1
            j-=1
        return True
    
    n = len(s)
    temp = []
    ans = []

    def dfs(start):
        if start==n:
            ans.append(list(temp))
            return 
        
        for i in range(start,n):
            if not isPalindrome(start,i): continue
            temp.append(s[start:i+1])
            dfs(i+1)
            temp.pop()
    
    dfs(0)
    return ans
```
```py
def partition(self, s: str) -> List[List[str]]:

    n = len(s)
    dp = [[True] * n for _ in range(n)]

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            dp[i][j] = (s[i] == s[j]) and dp[i + 1][j - 1]

    temp = []
    ans = []
    
    def dfs(start):
        if start==n:
            ans.append(list(temp))
            return 
        
        for i in range(start,n):
            if not dp[start][i]: continue
            temp.append(s[start:i+1])
            dfs(i+1)
            temp.pop()
    
    dfs(0)
    return ans
```
**93. 复原 IP 地址**
```py

def restoreIpAddresses(self, s: str) -> List[str]:
    n = len(s)
    if n<4: return []
    ans = []

    def dfs(x,k,tStr):
        if x==n: return 
        if k==3:
            if s[x]=='0':
                if x==n-1: 
                    ans.append(tStr+'0')
                return 
            if x<n-3: return
            val = int(s[x:n])
            if val<=255: ans.append(tStr+s[x:n])
            return 
        
        if s[x]=='0':
            dfs(x+1,k+1,tStr+"0.")
            return 
        for i in range(x,x+3):
            if i>=n: return 
            val = int(s[x:i+1])
            if val<=255: dfs(i+1,k+1,tStr+s[x:i+1]+'.')
            else: return
    
    dfs(0,0,"")
    return ans
```
```py
def restoreIpAddresses(self, s: str) -> List[str]:
    n = len(s)
    if n<4: return []
    ans = []

    def dfs(start,count,t):
        if start>=n: return
        if count==1:
            if start<n-3: return 
            
            if s[start]=='0':
                if start==n-1: ans.append(t+'0')
                return
            else:
                if start==n-3 and int(s[n-3:n])>255:
                    return
                ans.append(t+s[start:n])
            return 

        if s[start]=='0':
            dfs(start+1,count-1,t+'0.')
            return 

        for i in range(start,start+3):
            if i>=n-1: return 
            if i==start+2 and int(s[start:i+1])>255: return
            dfs(i+1,count-1,t+s[start:i+1]+'.')

    dfs(0,4,"")
    return ans
```
**78. 子集**
```py
def subsets(self, nums: List[int]) -> List[List[int]]:
    n = len(nums)
    ans = []
    temp = []

    def dfs(i):
        if i==n: 
            ans.append(list(temp))
            return 

        dfs(i+1)
        temp.append(nums[i])
        dfs(i+1)
        temp.pop()
    
    dfs(0)
    return ans
```
输出结果：

    [[],[3],[2],[2,3],[1],[1,3],[1,2],[1,2,3]]

```py
def subsets(self, nums: List[int]) -> List[List[int]]:

        temp = []
        ans = []
        n = len(nums)

        def dfs(x):
            ans.append(list(temp))
            if x==n: return 
  
            for i in range(x,n):
                temp.append(nums[i])
                dfs(i+1)
                temp.pop()
            
        dfs(0)
        return ans
```
输出结果：

    [[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]
**90. 子集 II**
```py
def subsetsWithDup(self, nums: List[int]):
    n = len(nums)
    temp = []
    ans = []
    nums.sort()

    def dfs(i):
        ans.append(list(temp))
        if i==n: return 

        for x in range(i,n):
            if x>i and nums[x]==nums[x-1]:continue
            temp.append(nums[x])
            dfs(x+1)
            temp.pop()
    dfs(0)
    return ans
```
**491. 非递减子序列**
```py
def findSubsequences(self, nums: List[int]):
    n = len(nums)
    temp = []
    ans = []

    def dfs(x,k):
        if k>=2: ans.append(list(temp))
        if x==n: return 

        seen = set()
        for i in range(x,n):
            if nums[i] in seen:continue
            if not temp or nums[i]>=temp[-1]:
                seen.add(nums[i])
                temp.append(nums[i])
                dfs(i+1,k+1)
                temp.pop()
        
    dfs(0,0)
    return ans
```
**46. 全排列**
```py
def permute(self, nums: List[int]) -> List[List[int]]:

    ans = []
    n = len(nums)

    def dfs(x):
        if x==n-1:
            ans.append(list(nums))
            return 
        
        for i in range(x,n):
            nums[x],nums[i] = nums[i],nums[x]
            dfs(x+1)
            nums[x],nums[i] = nums[i],nums[x]
    
    dfs(0)
    return ans
```
**47. 全排列 II**
```py
def permuteUnique(self, nums: List[int]) -> List[List[int]]:
    n = len(nums)
    ans = []
    # 不能采用排序，因为后面换了位置就又无序了

    def dfs(level):
        if level==n-1:
            ans.append(list(nums))
            return 
        
        seen = set()
        for i in range(level,n):
            if nums[i] in seen: continue
            seen.add(nums[i])
            nums[level],nums[i]=nums[i],nums[level]
            dfs(level+1)
            nums[level],nums[i]=nums[i],nums[level]
    
    dfs(0)
    return ans
```
**332. 重新安排行程**
```py
def findItinerary(self, tickets: List[List[str]]):
    def dfs(cur):
        while vec[cur]:
            tmp = heapq.heappop(vec[cur])
            dfs(tmp)
        stack.append(cur)
    
    vec = defaultdict(list)
    for depart,arrive in tickets:
        vec[depart].append(arrive)
    for key in vec:
        heapq.heapify(vec[key])
    
    stack = list()
    dfs("JFK")
    return stack[::-1]
```
**51. N-皇后**
```py
def solveNQueens(self, n: int) -> List[List[str]]:

    ans = []
    cols = [0] * n

    def valid(r, c):
        for pr in range(r):
            pc = cols[pr]
            if r+c==pr+pc or r-c==pr-pc:
                return False
        return True
        
    def dfs(r, s):
        if r==n:
            ans.append(['.'*c+'Q'+'.'*(n-1-c) for c in cols])
            return 
        for c in s:
            if valid(r, c): 
                cols[r] = c
                dfs(r+1, s-{c})
    
    dfs(0,set(range(n)))
    return ans
```
```py
def solveNQueens(self, n: int) -> List[List[str]]:

    ans = []
    cols = [0] * n
    can_use = [True]*n
    m = 2*n-1
    diag1 = [True]*m    # for r+c
    diag2 = [True]*m    # for r-c

    def dfs(r):
        if r==n:
            ans.append(['.'*c+'Q'+'.'*(n-1-c) for c in cols])
            return 
        for c in range(n):
            if can_use[c] and diag1[r+c] and diag2[r-c]:
                can_use[c]=diag1[r+c]=diag2[r-c]=False
                cols[r]=c
                dfs(r+1)
                can_use[c]=diag1[r+c]=diag2[r-c]=True
    dfs(0)
    return ans
```
**37. 解数独**
```py
def solveSudoku(self, board: List[List[str]]) -> None:
    """
    Do not return anything, modify board in-place instead.
    """
    rows = [[True]*9 for _ in range(9)]
    cols = [[True]*9 for _ in range(9)]
    block = [[[True]*9 for _ in range(3)] for _ in range(3)]

    valid = False
    spaces = []
    for i in range(9):
        for j in range(9):
            if board[i][j]=='.':
                spaces.append((i,j))
            else:
                val = int(board[i][j])-1
                rows[i][val]=cols[j][val]=block[i//3][j//3][val]=False
    
    bound = len(spaces)
    def dfs(pos):
        nonlocal valid
        if pos == bound:
            valid = True
            return 
        
        i,j = spaces[pos]
        for d in range(9):
            if rows[i][d] and cols[j][d] and block[i//3][j//3][d]:
                board[i][j]=str(d+1)
                rows[i][d]=cols[j][d]=block[i//3][j//3][d]=False
                dfs(pos+1)
                rows[i][d]=cols[j][d]=block[i//3][j//3][d]=True
            if valid: return
    dfs(0)
```
### 贪心算法
**455. 分发饼干**
```py
def findContentChildren(self, g: List[int], s: List[int]) -> int:
    g.sort()
    s.sort()
    gn,sn = len(g),len(s)

    i,j = 0,0

    while i<gn and j<sn:
        if s[j]>=g[i]:
            i+=1
        j+=1
    return i
```
**376.  摆动序列**
```py
class Solution:
def wiggleMaxLength(self, nums: List[int]) -> int:
    n = len(nums)
    if n==1: return 1

    ans = 1
    trend = 0
    for i in range(1,n):
        if nums[i]>nums[i-1] and trend<=0:
            ans += 1
            trend = 1
        elif nums[i]<nums[i-1] and trend>=0:
            ans+=1
            trend=-1
    
    return ans
```
```py
def wiggleMaxLength(self, nums: List[int]) -> int:
    trend = 0
    ans = 1

    for i,x in enumerate(nums[1:],1):
        if x>nums[i-1] and trend<=0:
            ans+=1
            trend = 1
        elif x<nums[i-1] and trend>=0:
            ans+=1
            trend = -1
    return ans
```
**53. 最大子数组和**
```py
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        pre  = nums[0]
        ans = pre

        for i in range(1,len(nums)):
            sums = nums[i]+max(0,pre)
            pre = sums
            ans = max(ans,sums)
        return ans
```
```py
def maxSubArray(self, nums: List[int]) -> int:
    pre = nums[0]
    ans = nums[0]

    for x in nums[1:]:
        if pre<=0:
            pre = x
        else: pre+=x
        if pre>ans: ans = pre
    return ans
```
**122. 买卖股票的最佳时机 II**
```py
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        ans = 0
        for i in range(1,len(prices)):
            ans+=max(0,prices[i]-prices[i-1])
        return ans
```
**55. 跳跃游戏**
```py
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        if n==1: return True

        # most can not reach i will fail

        most = 0
        for i in range(n-1):
            if most<i: return False
            most = max(most,i+nums[i])
            if most>=n-1: return True
        return False
```
```py
def canJump(self, nums: List[int]) -> bool:
    i,most = 0,0
    n = len(nums)

    while most<n-1:
        if i<=most: most = max(most,i+nums[i])
        else: return False
        i+=1
    return True
```

**45. 跳跃游戏 II**
```py
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n==1 or n==2: return n-1

        ans,most,end = 0,0,0

        for i in range(n-1):
            most = max(most,i+nums[i])
            if i==end:
                ans+=1
                end = most
        return ans
```

举例：

    nums = [2,3,5,1,4]

```py
def jump(self, nums: List[int]) -> int:
    ans = 0
    i,end,most = 0,0,0
    n = len(nums)

    for i in range(n-1):
        most = max(most,i+nums[i])
        if i==end:
            ans += 1
            if most>=n-1: break
            end = most
    return ans
```
**1005. K 次取反后最大化的数组和**
```py
def largestSumAfterKNegations(self, nums, k: int) -> int:
    oneMin = 101
    nums.sort()
    for i in range(len(nums)):
        if nums[i]<0:
            if k:
                nums[i] = -nums[i]
                k-=1
                oneMin = min(oneMin,nums[i])
        elif nums[i]==0: oneMin = 0
        else:
            oneMin = min(oneMin,nums[i])
    
    if not k or k%2==0: return sum(nums)
    else: return sum(nums)-2*oneMin
```
**134. 加油站**
```py
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost) -> int:
        if sum(gas)<sum(cost): return -1

        start = 0
        total = 0

        for i in range(len(gas)):
            total+=gas[i]-cost[i]
            if total<0:
                total = 0
                start = i+1
        return start
```
**860. 柠檬水找零**
```py
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five = 0
        ten = 0

        for b in bills:
            if b==5: five+=1
            elif b==10:
                if five: 
                    five-=1
                    ten+=1
                else: return False
            elif b==20:
                if ten and five:
                    ten-=1
                    five-=1
                    continue
                if five>=3:
                    five-=3
                else: return False
        return True
```
**406. 根据身高重建队列**
```py
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x:(-x[0],x[1]))
        ans = []
        for person in people:
            ans[person[1]:person[1]] = [person]
        return ans
```
**452. 用最少数量的箭引爆气球**
```py
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda x:x[1])
        right = points[0][1]
        ans = 1
        
        for i in range(1,len(points)):
            if points[i][0]>right:
                ans+=1
                right = points[i][1]
        return ans
```
```py
def findMinArrowShots(self, points) -> int:
    points.sort(key=lambda x: x[1])
    ans = 1
    right = points[0][1]

    for l,r in points[1:]:
        if l>right:
            ans+=1
            right = r
    return ans
```
**435. 无重叠区间**
```py
class Solution:
def eraseOverlapIntervals(self, intervals) -> int:
    intervals.sort(key=lambda x:x[1])
    ans = 1
    right = intervals[0][1]

    for i in range(1,len(intervals)):
        if intervals[i][0]>=right:
            ans+=1
            right=intervals[i][1]
    return len(intervals)-ans
```
```py
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda x: x[1])
        ans = 1
        right = intervals[0][1]

        for l,r in intervals[1:]:
            if l>=right:
                ans+=1
                right = r
        return len(intervals)-ans
```
**763. 划分字母区间**
```py
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        dic = dict()

        for i,c in enumerate(s): dic[c]=i

        most = 0
        ans = []
        start = -1
        for i,c in enumerate(s):
            most = max(most,dic[c])
            if i==most:
                ans.append(i-start)
                start = i
        return ans
```
**56. 合并区间**
```py
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda x:(x[0],x[1]))
        ans = [intervals[0]]

        for i in range(1,len(intervals)):
            if intervals[i][0]<=ans[-1][1]:
                ans[-1][1] = max(intervals[i][1],ans[-1][1])
            else: ans.append(list(intervals[i]))
        return ans

```
**738. 单调递增的数字**
```py
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        if n<=9: return n

        cacheN = str(n)
        strN = list(cacheN)
        n = len(strN)

        for i in range(1,n):
            if strN[i]<strN[i-1]: break
        
        if i==n: return n
        
        while i>0 and strN[i]<strN[i-1]:
            strN[i-1] = str(int(strN[i-1])-1)
            i-=1
        
        i+=1
        while i<n:
            strN[i]='9'
            i+=1
        return int(''.join(strN))
```
```py
def monotoneIncreasingDigits(self, n: int) -> int:
    strN = list(str(n))
    lens = len(strN)

    i = 1
    while i<lens:
        if strN[i]<strN[i-1]: break
        i+=1
    
    if i==lens: return n

    while i>=1 and strN[i]<strN[i-1]:
        strN[i-1] = chr(ord(strN[i-1])-1)
        i-=1
    i+=1
    strN[i:] = ['9']*(lens-i)
    return int(''.join(strN))
```
### 动态规划
**509. 斐波那契数**
```py
class Solution:
    def fib(self, n: int) -> int:
        if n<=1: return n

        a,b = 0,1
        for i in range(1,n):
            a,b = b,b+a
        return b
```
**70. 爬楼梯**
```py
class Solution:
    def climbStairs(self, n: int) -> int:
        if n<=2: return n

        a,b = 1,2
        for i in range(2,n):
            a,b = b,a+b
        return b
```
**746. 使用最小花费爬楼梯**
```py
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        a,b = 0,0
        n = len(cost)
        for i in range(2,n+1):
            a,b = b,min(a+cost[i-2],b+cost[i-1])
        return b

```
**62. 不同路径**
```py
def uniquePaths(self, m: int, n: int) -> int:
    dp = [[1]*n for _ in range(m)]
    for i in range(1,m):
        for j in range(1,n):
            dp[i][j]=dp[i-1][j]+dp[i][j-1]
    return dp[m-1][n-1]
```
```py
def uniquePaths(self, m: int, n: int) -> int:
    dp = [1]*n
    for i in range(1,m):
        for j in range(1,n):
            dp[j]=dp[j]+dp[j-1]
    return dp[n-1]
```
**63. 不同路径 II**
```py
def uniquePathsWithObstacles(self, obstacleGrid) -> int:
    if obstacleGrid[0][0]==1: return 0
    m,n = len(obstacleGrid),len(obstacleGrid[0])
    dp = [[0]*n for _ in range(m)]
    dp[0][0]=1
    for j in range(1,n):
        if obstacleGrid[0][j]: break
        else: dp[0][j]=1
    for i in range(1,m):
        if obstacleGrid[i][0]: break
        else: dp[i][0]=1
    
    for i in range(1,m):
        for j in range(1,n):
            if obstacleGrid[i][j]: dp[i][j]=0
            else: dp[i][j]=dp[i-1][j]+dp[i][j-1]
    return dp[m-1][n-1]
```
```py
def uniquePathsWithObstacles(self, obstacleGrid) -> int:
    if obstacleGrid[0][0]==1: return 0
    m,n = len(obstacleGrid),len(obstacleGrid[0])
    dp = [0]*n
    dp[0]=1
    for j in range(1,n):
        if obstacleGrid[0][j]: break
        else: dp[j]=1
    for i in range(1,m):
        if dp[0] and obstacleGrid[i][0]:
            dp[0]=0      
        for j in range(1,n):
            if obstacleGrid[i][j]: dp[j]=0
            else: dp[j]+=dp[j-1]
    return dp[n-1]
```
**343. 整数拆分**
```py
class Solution:
    def integerBreak(self, n: int) -> int:
        if n<=3: return n-1

        a = 0
        while n>3:
            if n==4: return pow(3,a)*n
            n-=3
            a+=1
        return pow(3,a)*n if n else pow(3,a)
```
**96. 不同的二叉搜索树**
```py
class Solution:
    def numTrees(self, n: int) -> int:
        @cache
        def dfs(l,r):
            if l>=r: return 1
            ans = 0
            for i in range(l,r+1):
                ans+=dfs(l,i-1)*dfs(i+1,r)
            return ans
        return dfs(1,n)
```
```py
def numTrees(self, n: int) -> int:
    # dp[n] = dp[0]*dp[n-1]+...+dp[n-1]*dp[0]
    dp = [0]*(n+1)
    dp[0]=dp[1]=1
    for i in range(2,n+1):
        for j in range(i):
            dp[i]+=dp[j]*dp[i-j-1]
    return dp[n]
```
**416. 分割等和子集**
```py
def canPartition(self, nums: List[int]) -> bool:
    sum_val = sum(nums)
    if sum_val&1: return False
    half = sum_val//2
    n = len(nums)

    dp = [[False]*(half+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=True

    for i in range(1,n+1):
        for j in range(1,half+1):
            if j>=nums[i-1]: 
                dp[i][j]=(dp[i-1][j] or dp[i-1][j-nums[i-1]])
            else: dp[i][j] = dp[i-1][j]
    return dp[n][half]
```
```py
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        sums = sum(nums)
        if sums&1: return False

        half = sums//2
        dp = [False]*(half+1)
        dp[0] = True

        for i in range(1,len(nums)+1):
            for j in range(half,0,-1):
                if j>=nums[i-1]:
                    dp[j] = dp[j] or dp[j-nums[i-1]]
        return dp[half]
```
```py
def canPartition(self, nums: List[int]) -> bool:
    sums = sum(nums)
    if sums&1: return False
    half = sums//2
    n = len(nums)
    dp = [False]*(half+1)
    dp[0]=True
    for i in range(1,n+1):
        for j in range(half,nums[i-1]-1,-1):
            dp[j]|=dp[j-nums[i-1]]
    return dp[half]
```
**1049. 最后一块石头的重量 II**
```py
def lastStoneWeightII(self, stones: List[int]) -> int:
    sums = sum(stones)
    tar = sums//2
    n = len(stones)
    dp = [[0]*(tar+1) for _ in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,tar+1):
            if j>=stones[i-1]:
                dp[i][j]=max(dp[i-1][j],stones[i-1]+dp[i-1][j-stones[i-1]])
            else: dp[i][j] = dp[i-1][j]
    return sums-2*dp[n][tar]
```
```py
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        # min = (sums-neg)-neg
        # min = sums-2*neg
        # 容量在无限接近sums//2时的最大价值

        sums = sum(stones)
        neg = sums//2
        dp = [0]*(neg+1)

        for i in range(1,len(stones)+1):
            for j in range(neg,0,-1):
                if j>=stones[i-1]:
                    dp[j]=max(dp[j],dp[j-stones[i-1]]+stones[i-1])
        return sums-2*dp[neg]
```
```py
def lastStoneWeightII(self, stones: List[int]) -> int:
    sums = sum(stones)
    tar = sums//2
    n = len(stones)
    dp = [0]*(tar+1)
    for i in range(1,n+1):
        for j in range(tar,stones[i-1]-1,-1):
            dp[j]=max(dp[j],stones[i-1]+dp[j-stones[i-1]])
    return sums-2*dp[tar]
```
**494. 目标和**
```py
def findTargetSumWays(self, nums: List[int], target: int) -> int:
    sums = sum(nums)
    neg = sums - target
    if neg&1: return 0
    neg //=2
    if neg<0 or neg>sum(nums): return 0
    n = len(nums)
    dp = [[0]*(neg+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0]=1
    for i in range(1,n+1):
        for j in range(neg+1):
            if j>=nums[i-1]:
                dp[i][j]=dp[i-1][j]+dp[i-1][j-nums[i-1]]
            else: dp[i][j]=dp[i-1][j]
    return dp[n][neg]
```
```py
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # tar = (sums-neg)-neg
        # neg = (sums-tar)/2

        if target<0: target=-target
        temp = sum(nums)-target
        if temp<0 or temp&1: return 0
        
        neg = temp//2
        dp = [0]*(neg+1)
        dp[0]=1

        for i in range(1,len(nums)+1):
            for j in range(neg,-1,-1):
                if j>=nums[i-1]:
                    dp[j]+=dp[j-nums[i-1]]
        return dp[neg]
```
```py
def findTargetSumWays(self, nums: List[int], target: int) -> int:
    sums = sum(nums)
    neg = sums - target
    if neg&1: return 0
    neg //=2
    if neg<0 or neg>sum(nums): return 0
    n = len(nums)
    dp = [0]*(neg+1)
    dp[0]=1
    for i in range(1,n+1):
        for j in range(neg,nums[i-1]-1,-1):
            dp[j]+=dp[j-nums[i-1]]
    return dp[neg]
```
**474. 一和零**
```py
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        strsN = [(x.count('0'),x.count('1')) for x in strs]
        
        dp = [[0]*(n+1) for _ in range(m+1)]

        for k in range(len(strs)):
            for i in range(m,-1,-1):
                for j in range(n,-1,-1):
                    if i>=strsN[k][0] and j>=strsN[k][1]:
                        dp[i][j]=max(dp[i][j], \
                        1+dp[i-strsN[k][0]][j-strsN[k][1]])
        return dp[m][n]
```
```py
def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
    dp = [[0]*(n+1) for _ in range(m+1)]
    for k in range(len(strs)):
        zeros,ones = strs[k].count('0'),strs[k].count('1')
        for i in range(m,zeros-1,-1):
            for j in range(n,ones-1,-1):
                dp[i][j]=max(1+dp[i-zeros][j-ones],dp[i][j])
    return dp[m][n]
```
**518. 零钱兑换 II**
```py
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:

        dic = dict()
        def dfs(i,a):
            if a==0: return 1
            if i==0: return 0

            if (i,a) in dic: return dic[(i,a)]

            if a>=coins[i-1]:
                dic[(i,a)] = dfs(i-1,a)+dfs(i,a-coins[i-1])
            else: dic[(i,a)] = dfs(i-1,a)
            return dic[(i,a)]
        return dfs(len(coins),amount)

```
**377. 组合总和 Ⅳ**
```py
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0]*(target+1)
        dp[0]=1

        for i in range(1,target+1):
            for num in nums:
                if num>i: continue
                else:
                    dp[i]+=dp[i-num]
        return dp[target]
```
**322. 零钱兑换**
```py
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount==0: return 0
        dp = [amount+1]*(amount+1)
        coins.sort()
        dp[0]=0
        for i in range(1, amount+1):
            for c in coins:
                if c<=i:
                    dp[i] = min(dp[i], 1+dp[i-c])
                else: break
        return dp[amount] if dp[amount]<=amount else -1
```
**279. 完全平方数**
```py
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [i for i in range(n+1)]

        for i in range(2,n+1):
            for j in range(1,int(i**0.5)+1):
                dp[i]=min(dp[i],1+dp[i-j*j])
        return dp[-1]
```
**139. 单词拆分**
```py
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        setW = set(wordDict)

        dp = [False]*(len(s)+1)
        dp[0]=True

        for i in range(1,len(s)+1):
            for j in range(i):
                if dp[j] and s[j:i] in setW:
                    dp[i]=True
                    break
        return dp[len(s)]
```
**198. 打家劫舍**
```py
class Solution:
    def rob(self, nums: List[int]) -> int:
        # dp[i] = max(dp[i-2]+nums[i],dp[i-1])
        # dp[0],dp[1]=nums[0],max(nums[0],nums[1])

        pre,cur = 0,0
        for num in nums:
            pre,cur = cur,max(pre+num,cur)
        return cur
```
**213. 打家劫舍 II**
```py
class Solution:
    def rob(self, nums: List[int]) -> int:

        def robA(house):
            pre,cur=0,0
            for h in house:
                pre,cur=cur,max(pre+h,cur)
            return cur
        
        return max(robA(nums[:-1]),robA(nums[1:])) if len(nums)!=1 else nums[0]
```
**337. 打家劫舍 III**
```py
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if not node: return 0,0
            lc,ln = dfs(node.left)
            rc,rn = dfs(node.right)
            no = max(lc,ln)+max(rc,rn)
            yes = ln+rn+node.val
            return yes,no
        return max(dfs(root))
        
```
**121. 买卖股票的最佳时机**
```py
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        most = 0
        cost = prices[0]

        for p in prices[1:]:
            most = max(most,p-cost)
            if p<cost: cost=p
        return most
```
**309. 买卖股票的最佳时机含冷冻期**
```py
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        # dp[i][0]=max(dp[i-1][0],dp[i-1][1]+prices[i]-fee)
        # dp[i][1]=max(dp[i-1][1],dp[i-1][0]-prices[i])
        # edge: dp[0][0]=0,dp[0][1]=-prices[0]

        sell,buy = 0,-prices[0]
        for p in prices[1:]:
            sell,buy=max(sell,buy+p-fee),max(buy,sell-p)
        return sell
```
**714. 买卖股票的最佳时机含手续费**
```py
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # dp[i]: 以i结尾最长严格递增子序列长度
        # dp[i] = max(dp[j])+1  0<=j<i and nums[j]<nums[i]

        # new dp
        # dp[i]: 长度为i的结尾值
        dp = [nums[0]]
        res = 1

        for num in nums[1:]:
            if num>dp[-1]:
                dp.append(num)
                res+=1
                continue
            else:
                i,j=0,res
                while i<=j:
                    mid=(i+j)//2
                    if dp[mid]<num: i=mid+1
                    else: j=mid-1
                dp[j+1]=num
        return res
        
```
**300. 最长递增子序列**
```py
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        # dp = [1]*len(nums)
        # for i in range(1,len(nums)):
        #     if nums[i]>nums[i-1]:
        #         dp[i]=dp[i-1]+1
        # return max(dp)

        ans = 1
        t = 1

        for i,num in enumerate(nums[1:],1):
            if num>nums[i-1]: 
                t+=1
                ans=max(ans,t)
            else: t=1
        return ans
```
**674. 最长连续递增序列**
```py
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        # dp[i][j]表示i，j开头最长结果
        # dp[i][j]=dp[i+1][j+1]+1 if s[i]==s[j] else 0
        # edge: dp[m-1][j],dp[i][n-1]

        # dp[i][j]: i,j结尾最长子串长度
        # dp[i][j] = dp[i-1][j-1]+1 if s[i]==t[j]
        # else: dp[i][j] = 0
        # return max(dp[i][j])
        # edge: dp[0][j],dp[i][0]

        n1,n2=len(nums1),len(nums2)
        dp = [[0]*n2 for _ in range(n1)]
        ans = 0

        for j in range(n2): 
            if nums1[0]==nums2[j]: 
                dp[0][j]=1
                ans = 1
        for i in range(1,n1): 
            if nums1[i]==nums2[0]: 
                dp[i][0]=1
                ans = 1

        for i in range(1,n1):
            for j in range(1,n2):
                if nums1[i]==nums2[j]: 
                    dp[i][j]=dp[i-1][j-1]+1
                    ans = max(ans,dp[i][j])
        return ans
```
**718. 最长重复子数组**
```py
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # dp[i][j]: 以i,j结尾最长子序列
        # return max(dp[i][j])
        # dp[i][j] = dp[i-1][j-1]+1 if s[i]==t[j]
        # dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        # edge: dp[i][0] dp[0][j]

        m,n = len(text1),len(text2)
        dp = [[0]*(n+1) for _ in range(m+1)]

        for i in range(1,m+1):
            for j in range(1,n+1):
                if text1[i-1]==text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]+1
                else: dp[i][j]=max(dp[i-1][j],dp[i][j-1])
        return dp[m][n]
                
```
**1143. 最长公共子序列**
```py
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # dp[i][j]: 以i,j结尾最长子序列
        # return max(dp[i][j])
        # dp[i][j] = dp[i-1][j-1]+1 if s[i]==t[j]
        # dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        # edge: dp[i][0] dp[0][j]

        m,n = len(text1),len(text2)
        dp = [[0]*(n+1) for _ in range(m+1)]

        for i in range(1,m+1):
            for j in range(1,n+1):
                if text1[i-1]==text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]+1
                else: dp[i][j]=max(dp[i-1][j],dp[i][j-1])
        return dp[m][n]
                
```
**1035. 不相交的线**
```py
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        m,n = len(nums1),len(nums2)

        dp = [0]*(n+1)
        for i in range(1,m+1):
            leftup=dp[0]
            for j in range(1,n+1):
                temp = dp[j]
                if nums1[i-1]==nums2[j-1]:
                    dp[j]=leftup+1
                else: dp[j]=max(dp[j],dp[j-1])
                leftup = temp
        return dp[n]
```
**392. 判断子序列**
```py
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        i,j=0,0

        while i<len(s) and j<len(t):
            if s[i]==t[j]:
                i+=1
            j+=1
        return i==len(s)
```
**583. 两个字符串的删除操作**
```py
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m,n = len(word1),len(word2)
        dp = [[0]*(n+1) for _ in range(m+1)]

        for i in range(1,m+1):
            for j in range(1,n+1):
                if word1[i-1]==word2[j-1]:
                    dp[i][j]=dp[i-1][j-1]+1
                else:
                    dp[i][j]=max(dp[i-1][j],dp[i][j-1])
        return m+n-2*dp[m][n]
```
**72. 编辑距离**
```py
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m,n = len(word1),len(word2)

        # dp = [[0]*(n+1) for _ in range(m+1)]
        # for j in range(n+1): dp[0][j]=j
        # for i in range(1,m+1): dp[i][0]=i

        # for i in range(1,m+1):
        #     for j in range(1,n+1):
        #         if word1[i-1]==word2[j-1]:
        #             dp[i][j]=dp[i-1][j-1]
        #         else:
        #             dp[i][j]=1+min(dp[i-1][j],dp[i-1][j-1],dp[i][j-1])
        # return dp[m][n]

        dp = list(range(n+1))

        for i in range(1,m+1):
            leftup = dp[0]
            dp[0]=i
            for j in range(1,n+1):
                temp = dp[j]
                if word1[i-1]==word2[j-1]:dp[j]=leftup
                else:dp[j]=1+min(dp[j-1],leftup,dp[j])
                leftup=temp
        return dp[-1]

```
**647. 回文子串**
```py
class Solution:
    def countSubstrings(self, s: str) -> int:
        # double pointer + center
        def extends(i,j):
            res = 0
            while i>=0 and j<n and s[i]==s[j]:
                res+=1
                i-=1
                j+=1
            return res
        
        ans,n = 0,len(s)
        for i in range(n): ans+=extends(i,i)
        for i in range(n-1): ans+=extends(i,i+1)
        return ans

```
**516. 最长回文子序列**
```py
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        # dp[i][j]为(i,j)之间的最大长度
        # dp[i][j] = dp[i+1][j-1]+2 if s[i]==s[j]
        # else dp[i][j] = max(dp[i+1][j], dp[i][j-1])

        n = len(s)
        dp = [[0]*n for _ in range(n)]
        for i in range(n-1,-1,-1):
            dp[i][i] = 1
            for j in range(i+1,n):
                if s[i]==s[j]:
                    dp[i][j] = dp[i+1][j-1]+2
                else: dp[i][j] = max(dp[i+1][j],dp[i][j-1])
        return dp[0][n-1]
```
### 单调栈
**739. 每日温度**
```py
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        ans = [0]*len(temperatures)
        stk = []
        for i,t in enumerate(temperatures):
            while stk and temperatures[stk[-1]]<t:
                ans[stk[-1]]=i-stk[-1]
                stk.pop()
            stk.append(i)
        return ans
                
```
**496. 下一个更大元素 I**
```py
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        dic = dict((num,i) for i, num in enumerate(nums1))
        stack=[]
        ans = [-1]*len(nums1)

        for i,num in enumerate(nums2):
            while stack and stack[-1]<num:
                if stack[-1] in dic:
                    ans[dic[stack[-1]]]=num
                stack.pop()
            stack.append(num)
        return ans
```
**503. 下一个更大元素 II**
```py
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [-1]*n

        stk = []
        i = 0
        while i<2*n-1:
            while stk and nums[stk[-1]]<nums[i%n]:
                ans[stk[-1]]=nums[i%n]
                stk.pop()
            stk.append(i%n)
            i+=1
        return ans
```
### 图论
**797. 所有可能的路径**
```py
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        ans = []
        temp = []

        def dfs(i):
            if i==len(graph)-1:
                ans.append(list(temp))
                return 
            for node in graph[i]:
                temp.append(node)
                dfs(node)
                temp.pop()
        temp.append(0)
        dfs(0)
        return ans
```
**200. 岛屿数量**
```py
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m,n = len(grid),len(grid[0])
        def dfs(i,j):
            grid[i][j]='0'
            for r,c in [(i-1,j),(i,j+1),(i+1,j),(i,j-1)]:
                if 0<=r<m and 0<=c<n and grid[r][c]=='1':
                    dfs(r,c)
        
        ans=0
        for r in range(m):
            for c in range(n):
                if grid[r][c]=='1':
                    ans+=1
                    dfs(r,c)
        return ans
        
```
**695. 岛屿的最大面积**
```py
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        
        m,n = len(grid),len(grid[0])
        def dfs(i,j):
            grid[i][j]=0
            count = 1
            for r,c in [(i-1,j),(i,j-1),(i,j+1),(i+1,j)]:
                if 0<=r<m and 0<=c<n and grid[r][c]==1:
                    count+=dfs(r,c)
            return count
        
        ans=0
        for r in range(m):
            for c in range(n):
                if grid[r][c]==1:
                    ans = max(ans,dfs(r,c))
        return ans
```
**1020. 飞地的数量**
```py
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        def dfs(i,j):
            grid[i][j]=0
            for r,c in [(i-1,j),(i,j-1),(i,j+1),(i+1,j)]:
                if 0<=r<m and 0<=c<n and grid[r][c]==1:
                    dfs(r,c)
        
        m,n = len(grid),len(grid[0])
        for i in range(m):
            if grid[i][0]==1: dfs(i,0)
            if grid[i][n-1]==1: dfs(i,n-1)
        
        for j in range(n):
            if grid[0][j]==1: dfs(0,j)
            if grid[m-1][j]==1: dfs(m-1,j)
        
        ans = 0
        for i in range(1,m-1):
            for j in range(1,n-1):
                if grid[i][j]==1: ans+=1
        return ans
```
**130. 被围绕的区域**
```py
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def dfs(i,j):
            board[i][j]='A'
            for r,c in [(i-1,j),(i,j-1),(i,j+1),(i+1,j)]:
                if 0<=r<m and 0<=c<n and board[r][c]=='O':
                    dfs(r,c)
        
        m,n = len(board),len(board[0])

        for i in range(m):
            if board[i][0]=='O': dfs(i,0)
            if board[i][n-1]=='O': dfs(i,n-1)
        for j in range(n):
            if board[0][j]=='O': dfs(0,j)
            if board[m-1][j]=='O': dfs(m-1,j)
        for i in range(m):
            for j in range(n):
                if board[i][j]=='A': board[i][j]='O'
                elif board[i][j]=='O': board[i][j]='X'
        
```
**417. 太平洋大西洋水流问题**
```py
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        setp,seta=set(),set()

        def dfs(i,j,setx):
            setx.add((i,j))
            for r,c in [(i-1,j),(i,j-1),(i,j+1),(i+1,j)]:
                if 0<=r<m and 0<=c<n and heights[r][c]>=heights[i][j]:
                    if (r,c) not in setx:
                        dfs(r,c,setx)
        
        m,n = len(heights),len(heights[0])
        for i in range(m):
            if (i,0) not in setp: dfs(i,0,setp)
            if (i,n-1) not in seta: dfs(i,n-1,seta)
        for j in range(n):
            if (0,j) not in setp: dfs(0,j,setp)
            if (m-1,j) not in seta: dfs(m-1,j,seta)
        
        return list(map(list,seta&setp))
```
**463. 岛屿的周长**
```py
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        ans = 0
        m,n = len(grid),len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    for r,c in [(i-1,j),(i,j-1),(i,j+1),(i+1,j)]:
                        if 0<=r<m and 0<=c<n:
                            if grid[r][c]==0: ans+=1
                        else: ans+=1
        return ans
```
**1971. 寻找图中是否存在路径**
```py
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        father = list(range(n))

        def find(i):
            if father[i]==i: return i
            else: 
                father[i]=find(father[i])
                return father[i]
        
        # def join(i,j):
        #     i,j = find(i),find(j)
        #     if i==j: return 
        #     else:father[i]=j

        for i,j in edges:
            father[find(i)]=find(j)
            if find(destination)==find(source): return True
        return find(destination)==find(source)

```
**684. 冗余连接**
```py
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        def find(i):
            if father[i]==i: return i
            father[i]=find(father[i])
            return father[i]
        
        father = list(range(len(edges)+1))
        for i,j in edges:
            if find(i)==find(j): return [i,j]
            father[find(i)]=find(j)

            
```


