package main
import "fmt"
import "sort"
import "os"

func  fibonacci(n int) int {
	if n < 0 {
		fmt.Println("Input must be a non-negative integer.")
		os.Exit(1)
	}
	if n == 0 {
		return 0
	} else if n == 1 {
		return 1
	} else {
		return fibonacci(n-1) + fibonacci(n-2)
	}
}

func binarySearch(arr []int, target int) int {
	left, right := 0, len(arr)-1

	for left <= right {
		mid := left + (right-left)/2
		if arr[mid] == target{
			return mid
		} else if arr[mid] < target {
			left = mid + 1
		}else{
			right = mid - 1
		}
	}
	return -1 // Target not found
}


type Book struct {
	name string
	author string
	year int
	price float64
	cate string
}

func main() {
	// Example usage of the function
	const n int = 5
	const target int = 3
	arr := []int{ 10,9,3,2,-1,4,5,2,1 }

	// m := make(map[string]int)

	// 创建一个初始容量为 10 的 Map
	m := make(map[string]int, 10)
	// 添加一些键值对
	m["one"] = 1
	m["two"] = 2
	m["three"] = 3
	m[""] = 0
	fmt.Printf("map array: %v\n", m)
	type SortBy []int
	
	// func (a SortBy) Len() int           { return len(a) }
	// func (a SortBy) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
	sort.Ints(arr)
	
	fmt.Printf("Sorted array: %v\n", arr)

	fmt.Printf("Slicing array: %v\n", arr[:5])

	var pow = []int{1, 2, 4, 8, 16, 32, 64, 128}
	 // 遍历 pow 切片，i 是索引，v 是值
   for i, v := range pow {
      // 打印 2 的 i 次方等于 v
      fmt.Printf("2**%d = %d\n", i, v)
   }

	index := binarySearch(arr, target)
	fmt.Printf("The index of %d in the sorted array is: %d\n", target, index)
	fmt.Printf("The %d-th Fibonacci number is: %d\n", n, fibonacci(n))

	// structure
	var book1 Book
	book1.name = "The Great Gatsby"
	book1.author = "F. Scott Fitzgerald"
	book1.year = 1925
	book1.price = 10.99
	book1.cate = "Novel"

	fmt.Printf("Book Details:\nName: %s\nAuthor: %s\nYear: %d\nPrice: $%.2f\nCategory: %s\n",
		book1.name, book1.author, book1.year, book1.price, book1.cate)

	// pointer example
	var book_pt *Book
	book_pt = &book1
	fmt.Printf("Book Details through pointer:\nName: %s\nAuthor: %s\nYear: %d\nPrice: $%.2f\nCategory: %s\n",
		book_pt.name, book_pt.author, book_pt.year, book_pt.price, book_pt.cate)


	book2 :=Book{ name: "To Kill a Mockingbird", author: "Harper Lee", year: 1960, price: 7.99, cate: "Novel"}

	fmt.Printf("\n\nBook2 Details:\nName: %s\nAuthor: %s\nYear: %d\nPrice: $%.2f\nCategory: %s\n",
		book2.name, book2.author, book2.year, book2.price, book2.cate)

}