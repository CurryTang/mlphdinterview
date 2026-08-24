# Effective Modern C++ 1 · 类型推导与 auto（条款 1-6）

这一讲对应 Scott Meyers《Effective Modern C++》第 1 章"Deducing Types"的六个条款。C++11 之后，`auto`、模板参数、`decltype` 三套类型推导规则贯穿了几乎所有现代 C++ 代码，但它们彼此并不完全一致：`auto` 大体上复用模板类型推导规则，却在花括号初始化上单开了一个例外；`decltype` 又完全是另一套规则，专门用来"精确复现某个表达式的类型"，包括它的引用性和 const 性。不熟悉这三套规则的边界，最常见的后果不是编译报错（那还算幸运），而是代码悄悄编译通过，但变量的实际类型和你以为的不一样——多一层引用、多一层 const、或者退化成了指针，调试起来非常隐蔽。这一章要解决的问题是：给定一段声明，能不能不看编译器输出就准确说出每个类型推导的结果。

```text
看到类型推导问题该检查什么：
1. 先分清这是"模板类型推导"（函数模板参数、auto 变量、auto 返回类型、lambda 参数）还是"decltype 推导"（decltype(x)、decltype(auto)）——两套规则不通用。
2. 是模板类型推导的话，先看形参声明是三种形式里的哪一种：指针/引用（非万能引用）、万能引用 T&&、按值传递。这决定了实参的引用性、const 性、数组/函数名会不会被保留。
3. 遇到 auto，先套用模板推导规则，再单独检查一件事：初始化表达式是不是花括号 { }。如果是，auto 会被特殊对待，推导成 std::initializer_list；同样的花括号放到模板参数推导的位置上则直接编译失败。
4. 遇到 auto 作为函数返回类型或 C++14 泛型 lambda 的参数类型，记住这里用的是模板类型推导规则而不是"auto 变量"那套规则，花括号初始化会失败。
5. 看到 decltype，先看操作数是不是"裸的变量名"：是裸名字就直接给出该变量的声明类型；如果套了一层括号（比如 decltype((x))）或者操作数本身是一个更复杂的左值表达式，规则切换成"给出该表达式类型的左值引用"，这是最容易被面试题当陷阱的点。
6. 不确定实际推导出的类型时，别猜——用编译器报错、typeid 或 Boost.TypeIndex 验证一遍，尤其是涉及引用折叠、cv 限定符、代理类型的场景。
7. 用 auto 声明变量后如果发现行为异常（不能取地址赋值、隐式转换代价大、值不稳定），先怀疑初始化表达式的返回类型是不是一个代理类型（proxy type），而不是怀疑 auto 本身出了错。
```

---

## 条款 1：理解模板类型推导

**核心结论**：函数模板 `template<typename T> void f(ParamType param);` 调用 `f(expr)` 时，编译器要同时推导 `T` 和 `ParamType`，推导规则完全由 `ParamType` 的形式决定，可以分成三种情况：`ParamType` 是指针或引用（但不是万能引用）、`ParamType` 是万能引用、`ParamType` 既不是指针也不是引用（按值传递）。这三种情况对"要不要保留 `expr` 的引用性、const/volatile 限定符"给出完全不同的答案，是后面理解 `auto` 推导规则的基础。

**情况一：`ParamType` 是指针或左值引用（非万能引用）**

推导时先忽略 `expr` 的引用性（如果它本身是个引用变量，只看它引用的那个对象的类型），再看这个类型能不能匹配 `ParamType`，从而分别推出 `T` 和完整的 `ParamType`。const 性会被保留。

```cpp
template<typename T>
void f(T& param) { }

int x = 27;
const int cx = x;
const int& rx = x;

f(x);   // T 推导为 int，         param 类型为 int&
f(cx);  // T 推导为 const int，   param 类型为 const int&   —— const 被保留
f(rx);  // T 推导为 const int，   param 类型为 const int&   —— rx 的引用性被剥离，只看被引用对象的 const 性
```

如果形参是 `const T&`，则 const 已经写死在 `ParamType` 里，`T` 本身就不会再带 const：

```cpp
template<typename T>
void f2(const T& param) { }

f2(x);   // T 推导为 int
f2(cx);  // T 推导为 int  —— cx 的 const 已经被 ParamType 自带的 const 吸收
```

**情况二：`ParamType` 是万能引用（`T&&`，且 `T` 由该次调用推导）**

这是唯一一种"实参是左值还是右值会影响推导结果"的情况：传入左值时，`T` 被推导为左值引用类型，`param` 的类型经过引用折叠变成左值引用；传入右值时，规则退化成和情况一一样，`T` 推导为非引用类型。

```cpp
template<typename T>
void f3(T&& param) { }

int x = 27;
const int cx = x;

f3(x);            // x 是左值，T 推导为 int&，  param 类型为 int&
f3(cx);           // cx 是左值，T 推导为 const int&
f3(27);           // 27 是右值，T 推导为 int，  param 类型为 int&&
```

注意 `T&&` 只有在“`T` 由本次调用推导”这个前提下才是万能引用；如果 `T` 是一个已经确定的具体类型（比如类模板的成员函数里用类的模板参数），`T&&` 就是普通右值引用，只能绑定右值，不适用这里的特殊规则。

**情况三：`ParamType` 既不是指针也不是引用（按值传递）**

`param` 是 `expr` 的一份完整拷贝，因此 `expr` 的 const、volatile、引用性统统与推导无关——它们描述的是原对象不能被怎样操作，而 `param` 是一个全新的、独立的对象，可以随意修改。

```cpp
template<typename T>
void f4(T param) { }

const int cx = 27;
const int& rx = cx;

f4(x);   // T 推导为 int
f4(cx);  // T 推导为 int —— const 被剥离
f4(rx);  // T 推导为 int —— const 和引用性都被剥离
```

**数组/函数名的退化规则**：数组和函数名在按值传递时会退化（decay）为指向首元素/函数的指针，但如果用来初始化一个引用类型的形参，则不会退化，而是正确绑定到数组或函数本身的类型。

```cpp
template<typename T>
void byValue(T param) { }        // 按值：数组退化为指针

template<typename T>
void byRef(T& param) { }         // 引用：保留数组类型

const char name[] = "quant";     // 类型是 const char[6]

byValue(name);   // T 推导为 const char*         —— 数组退化为指针，丢失了大小信息
byRef(name);     // T 推导为 const char (&)[6]   —— 引用正确绑定到数组类型，大小信息保留下来
```

```text
决策树：给定 template<typename T> void f(ParamType param); 调用 f(expr)
                     ┌─ ParamType 是 T& / T* / const T& 等（非万能引用）
                     │      → 忽略 expr 的引用性，保留 const/volatile
调用 f(expr) ────────┼─ ParamType 是 T&&（T 由本次调用推导，万能引用）
                     │      → expr 是左值：T 推导为左值引用；expr 是右值：按情况一处理
                     └─ ParamType 是 T（按值）
                            → 忽略 expr 的引用性、const、volatile，产生独立拷贝
        （数组 / 函数名：按值退化为指针，用于初始化引用则保留原类型）
```

**常见追问 / 面试陷阱**

> 面试官常问"`T&&` 一定是万能引用吗"。不是，只有在 `T` 本身是这次调用要推导的模板参数（或者是 `auto&&`）时才是万能引用；如果 `T` 是外层已经固定下来的类型（比如 `std::vector<T>::push_back(T&& val)` 里的 `T` 来自类模板参数而非该成员函数），`T&&` 是普通右值引用。判定标准是看这个 `&&` 出现处是否存在"针对这次调用的类型推导"。

**要记住**
- `ParamType` 是引用或指针（非万能引用）时：忽略实参引用性，保留 const/volatile。
- `ParamType` 是万能引用时：左值实参让 `T` 推导为左值引用，右值实参按常规规则处理。
- `ParamType` 按值传递时：无视实参的 const、volatile、引用性，`param` 是独立拷贝。
- 数组和函数名在按值传递语境下退化为指针；用它们初始化引用类型的形参则不退化。

---

## 条款 2：理解 auto 类型推导

**核心结论**：除了一个例外，`auto` 类型推导和条款 1 的模板类型推导完全等价——把 `auto` 当成模板参数 `T`，把变量的类型修饰符（`*`、`&`、`const` 等）当成 `ParamType`，套用条款 1 的三种情况即可；唯一的例外发生在用花括号初始化时。

```cpp
auto x = 27;         // 相当于按值传递情况：x 推导为 int
const auto cx = x;   // cx 推导为 const int
const auto& rx = x;  // rx 推导为 const int&

auto&& uref1 = x;    // x 是左值，uref1 推导为 int&   —— 万能引用情况
auto&& uref2 = 27;   // 27 是右值，uref2 推导为 int&&
```

**花括号初始化的例外**：`auto x = {1, 2, 3};` 会把 `x` 推导为 `std::initializer_list<int>`，这是 `auto` 独有的特殊规则——`auto` 看到花括号初始化时，先把它当成 `std::initializer_list`，再对里面的元素类型做模板推导。而如果把同样的花括号交给模板类型推导（不是通过 `auto`，而是显式的函数模板调用），编译器无法把一个花括号列表推导成任何具体的 `T`，直接编译失败。

```cpp
auto x1 = 27;         // int
auto x2(27);          // int
auto x3 = {27};       // std::initializer_list<int>
auto x4{27};          // C++17 起是 int；C++11/14 曾经也推导成 initializer_list<int>，是历史遗留的不一致点

template<typename T>
void f(T param) { }

// f({11, 23, 9});    // 编译失败：模板类型推导无法从花括号列表推出 T
```

**auto 作为返回类型或 C++14 泛型 lambda 参数时用的是模板类型推导规则，不是"auto 变量"规则**。这意味着花括号初始化的特殊待遇在这两个场景下完全不生效：

```cpp
auto f() {
    return {1, 2, 3};   // 编译失败：函数返回类型位置的 auto 走模板类型推导，
}                       // 无法从花括号推出返回类型

auto x = {1, 2, 3};     // 正确：x 是 std::initializer_list<int>，走的是"auto 变量"规则

auto lambda = [](auto param) { /* ... */ };
// lambda({1, 2, 3});   // 同样编译失败：泛型 lambda 的 auto 参数走模板类型推导
```

**要记住**
- `auto` 推导规则等价于模板类型推导，把 `auto` 类比成模板参数 `T` 即可。
- 唯一例外：`auto` 变量用花括号初始化时，一律推导为 `std::initializer_list<元素类型>`。
- 同样的花括号初始化放在模板类型推导上下文（普通函数模板参数）里会直接编译失败。
- `auto` 作为函数返回类型、或作为 C++14 泛型 lambda 的形参类型时，走的是模板类型推导规则而非 auto 变量规则，因此不能用花括号初始化触发 initializer_list 推导。

---

## 条款 3：理解 decltype

**核心结论**：对一个"裸变量名"用 `decltype`，得到的就是这个变量被声明时的确切类型，不做任何修饰或退化；但只要操作数不是单纯的名字，而是一个更复杂的左值表达式（哪怕只是在名字外面套一层括号），`decltype` 的规则就切换成"给出该表达式类型的左值引用"，这条切换规则是 `decltype` 里最容易被面试题当陷阱的地方。

```cpp
const int i = 0;
decltype(i) a = 0;     // a 的类型是 const int —— i 是裸名字，原样复现声明类型

struct Widget { double x; };
Widget w;
decltype(w.x) b = 0.0; // b 的类型是 double

int x = 0;
decltype(x) c = 0;     // c 的类型是 int      —— x 是裸名字
decltype((x)) d = c;   // d 的类型是 int&     —— (x) 是括号包裹的左值表达式，不再是"裸名字"
```

`decltype((x))` 这个例子值得单独拆开看：`(x)` 作为一个表达式，其求值结果是一个左值（可以取地址、可以出现在赋值号左边），根据 `decltype` 对"非名字的左值表达式"的规则，结果类型是该表达式类型的左值引用，也就是 `int&`，而不是 `int`。这条规则会导致一个真实的坑：

```cpp
decltype(auto) f1() {
    int x = 0;
    return x;      // decltype(x) 是 int，f1 按值返回，安全
}

decltype(auto) f2() {
    int x = 0;
    return (x);    // decltype((x)) 是 int&，f2 返回一个指向局部变量的引用！
}                  // x 在函数返回后被销毁，调用方拿到的是悬空引用，未定义行为
```

**`decltype(auto)`（C++14）的主要用途：让返回类型精确复现某个表达式应有的引用性和 const 性，而普通 `auto` 会把这些信息剥离掉**。典型场景是写一个"根据下标访问容器并返回引用"的转发函数：

```cpp
template<typename Container, typename Index>
auto authAndAccess(Container&& c, Index i) -> decltype(auto) {
    // authenticateUser();  // 假设这里有权限校验逻辑
    return std::forward<Container>(c)[i];
}
```

如果把返回类型改成普通 `auto`：

```cpp
template<typename Container, typename Index>
auto authAndAccessBad(Container&& c, Index i) {
    return std::forward<Container>(c)[i];   // auto 走模板类型推导：按值返回，引用性被剥离
}

std::vector<int> v{1, 2, 3};
// authAndAccessBad(v, 0) = 10;   // 编译失败：返回的是一份拷贝（右值），不能作为赋值目标
```

`operator[]` 对非 const 容器通常返回 `T&`，但普通 `auto` 按模板类型推导的"按值传递"规则处理返回表达式，会把这个引用性剥离，函数变成返回一份拷贝，调用方就无法通过返回值修改容器里的元素。改成 `decltype(auto)` 后，返回类型的推导改用 `decltype` 规则，对 `std::forward<Container>(c)[i]` 这个表达式而言就是它本身的类型（`T&`），引用性被完整保留，`authAndAccess(v, 0) = 10;` 才能正常工作。

**要记住**
- `decltype(变量名)`：原样给出该变量的声明类型，不做任何调整。
- `decltype(更复杂的左值表达式)`（包括仅仅加一层括号的 `decltype((x))`）：结果是该表达式类型的左值引用，这条规则常被当作陷阱题。
- `decltype(auto)`（C++14）按 `decltype` 规则推导，能保留 `auto` 会剥离掉的引用性和 const 性，常用于"返回容器元素引用"这类转发函数的返回类型。
- 写 `decltype(auto)` 返回类型时要小心不要在 `return` 语句里无意间多套一层括号，否则可能把按值返回的局部变量意外变成返回悬空引用。

---

## 条款 4：掌握查看推导类型的方法

**核心结论**：类型推导结果不应该靠猜，实践中有三类工具可用：IDE 的悬停提示（方便但对复杂类型经常显示不准确甚至出错）、故意触发的编译错误（把推导出的类型强行打印在编译器的诊断信息里，最可靠）、运行期工具 `typeid` 和 Boost.TypeIndex（前者信息有损，后者精确）。

**IDE 悬停提示**：多数编辑器把鼠标悬停在变量上就能看到推导类型，对简单情形足够用，但涉及模板嵌套、引用折叠、cv 限定符时，IDE 给出的往往是简化甚至错误的近似结果，不能完全依赖。

**故意制造编译错误**：让编译器把它推导出的类型直接打进报错信息里，是最不会说谎的办法。一个常见手法是声明一个只做声明不做定义的类模板，用它去实例化目标类型：

```cpp
template<typename T>
class TypeDisplayer;   // 只声明，不定义

template<typename T>
void f(const T& param) {
    TypeDisplayer<T> td;          // 故意实例化一个不完整类型，触发编译错误
    TypeDisplayer<decltype(param)> tp;  // 同时也能看到 param 的类型
}

int x = 0;
const int& rx = x;
f(rx);
// 编译器会报错，错误信息里通常会完整打印出 T 和 decltype(param) 被实例化成的具体类型，
// 例如提示 "TypeDisplayer<int>" 和 "TypeDisplayer<int const&>" 未定义
```

**运行期工具**：`typeid(x).name()` 能在运行期取到类型名字符串，但标准只保证它返回的是一个实现定义的字符串，很多编译器会把它做成"编译器内部的简化/mangled 名字"，并且这个结果会丢失 const 限定符和引用性等信息（比如一个 `const int&` 变量的 `typeid().name()` 打印结果常常就是 `int` 对应的名字）。

```cpp
#include <typeinfo>

template<typename T>
void f(const T& param) {
    std::cout << "T =     " << typeid(T).name() << '\n';
    std::cout << "param = " << typeid(param).name() << '\n';
    // 常见输出：两行结果一样（比如都是 "i"，代表 int），
    // 看不出 param 实际类型里的 const 和引用
}
```

如果需要精确保留 cv 限定符和引用性，可以用 Boost.TypeIndex 提供的 `boost::typeindex::type_id_with_cvr<T>().pretty_name()`，它专门设计用来避免 `typeid` 的信息丢失问题，能打印出类似 `int const&` 这样完整的类型描述。

**要记住**
- IDE 悬停提示速度快但不完全可靠，复杂类型要留个心眼。
- 故意触发编译错误（例如实例化一个未定义的类模板）是最可靠的静态查看手段，错误信息里会打印出真实的类型。
- `typeid(x).name()` 是运行期手段，但返回的字符串是实现定义的，且会丢失 const/引用等限定信息，不能当作精确判断依据。
- 需要精确保留 cv 限定符和引用性时用 `boost::typeindex::type_id_with_cvr<T>().pretty_name()`。

---

## 条款 5：优先使用 auto 而非显式类型声明

**核心结论**：`auto` 不只是少打几个字符的语法糖，它能系统性地消除几类常见错误——忘记初始化、类型名写错导致的隐式转换开销、重构时类型不同步——但代价是牺牲了在声明处直接看到类型这一点可读性。

**避免忘记初始化**：`auto` 变量必须有初始化表达式才能通过编译，天然杜绝了声明一个未初始化裸类型变量的可能性。

```cpp
int x;          // 合法但危险：x 是未初始化的，值不确定
auto y;         // 编译错误：auto 必须有初始化表达式，强制养成初始化的习惯
auto z = 0;     // 正确
```

**避免冗长的迭代器/lambda 类型声明**：

```cpp
std::map<std::string, std::vector<int>> data;

// 显式写法，冗长且容易和实际的 pair 类型 (const std::string, std::vector<int>) 对不上
for (std::map<std::string, std::vector<int>>::iterator it = data.begin(); it != data.end(); ++it) { }

// auto 写法：简洁，并且类型跟着 data.begin() 的真实返回类型走，不会写错
for (auto it = data.begin(); it != data.end(); ++it) { }

auto cmp = [](int a, int b) { return a < b; };  // lambda 的闭包类型只有编译器知道，只能用 auto 承接
```

**避免"类型偷懒"引入的隐式转换代价**：一个经典例子是把 `std::vector<int>::size_type`（通常是 `unsigned` 的某种别名，在多数 64 位平台上是 64 位无符号整数）随手写成 `unsigned` 或者 `float`：

```cpp
std::vector<int> v(1000000, 1);

float sz1 = v.size();      // size() 返回 size_type，被隐式转换/截断成 float，
                            // float 的有效精度大约只有 6-7 位十进制数字，
                            // 对于足够大的 size() 结果会产生精度丢失
for (unsigned i = 0; i < v.size(); ++i) { }
                            // 如果 size_type 比 unsigned 宽（常见于 64 位平台），
                            // v.size() 会被隐式截断，某些边界条件下可能导致比较结果出错或死循环

auto sz2 = v.size();        // sz2 被正确推导为 std::vector<int>::size_type，不存在这个问题
for (auto i = 0u; i < v.size(); ++i) { }   // 至少类型意图清晰，但真正安全的写法仍是 auto i = decltype(v.size()){0}
```

**让重构更安全**：如果某个函数的返回类型后续发生变化（比如把 `int` 改成 `int64_t`，或者把某个容器的 `value_type` 换掉），所有用 `auto` 承接返回值的调用点会自动跟着类型一起变化；而写了显式类型的调用点，要么因为类型不匹配触发隐式转换（编译能过但语义可能出问题），要么需要开发者手动逐处修改。

**已知限制**：`auto` 把变量的实际类型隐藏在了初始化表达式背后，阅读代码时无法在声明处直接看出类型，对复杂表达式尤其明显。常见的缓解办法是给变量取一个能清楚表达"是什么"的名字、依赖 IDE 悬停提示，或者在类型确实重要（比如接口边界）的地方仍然显式声明类型。

**要记住**
- `auto` 强制初始化，天然避免未初始化变量。
- `auto` 省去了迭代器、函数对象、lambda 闭包这类难以手写或根本写不出来的类型声明。
- `auto` 能避免"类型偷懒"造成的隐式转换/精度丢失问题，因为它直接复用表达式的真实类型。
- `auto` 让重构更安全：底层类型变化时，用 auto 声明的调用点自动同步，不需要逐处修改。
- 代价是牺牲了声明处的可读性，需要用命名规范或工具弥补。

---

## 条款 6：当 auto 推导的类型不符合要求时，使用显式类型初始化惯用法

**核心结论**：当初始化表达式返回的是一个"代理类型"（proxy type，一种设计上假装是另一种类型、但实际不是的类型，通常出于性能或实现原因存在）时，`auto` 会老老实实地把这个代理类型本身推导出来，而不是它假装成的那个类型；这经常导致悬空引用或其他未定义行为，解决办法是用 `static_cast` 强制在初始化处就完成到目标类型的转换，这就是"显式类型初始化惯用法"（explicitly typed initializer idiom）。

**经典陷阱：`std::vector<bool>`**。`std::vector<bool>` 是标准库里的一个特化，为了节省空间把每个 `bool` 压缩成一个 bit 存储，因此它没办法像 `std::vector<int>` 那样让 `operator[]` 返回一个真正的 `bool&`（C++ 不允许对单个 bit 取地址/引用），只能返回一个代理对象 `std::vector<bool>::reference`，这个代理对象内部通常持有一个指向底层字数据的指针加上位偏移，表现得像 `bool&` 但本质不是。

```cpp
std::vector<bool> features(const Widget& w) {
    return { true, false, true, true, false };   // 假设按业务逻辑构造一个临时 vector<bool>
}

Widget w;
auto highPriority = features(w)[5];
// highPriority 被推导为 std::vector<bool>::reference（代理类型），不是 bool
// features(w) 返回的是一个临时对象，这条语句结束后临时对象就被销毁；
// 而 highPriority 这个代理对象内部持有指向该临时对象底层数据的指针，
// 临时对象销毁后这个指针就是悬空的
processPriority(highPriority);   // 未定义行为：读取一个已经悬空的代理对象
```

问题的根源不是 `auto` 推导错了——`auto` 忠实地推导出了 `operator[]` 的真实返回类型 `std::vector<bool>::reference`，问题在于这个代理对象的生存期依赖于它所指向的底层数据（这里是一个已经销毁的临时 `vector<bool>`），而调用方以为自己拿到的是一个独立的 `bool` 值。

**修复：显式类型初始化惯用法**。在初始化处用 `static_cast` 强制触发代理类型向真实类型的转换，让转换在临时对象销毁之前就完成：

```cpp
auto highPriority = static_cast<bool>(features(w)[5]);
// static_cast<bool> 强制调用 std::vector<bool>::reference 到 bool 的转换运算符，
// 这次转换在 "features(w) 返回的临时 vector<bool>" 还存活的这条语句内完成，
// highPriority 拿到的是一份独立的 bool 值拷贝，不再依赖任何已经销毁的底层数据
```

**这个惯用法的适用范围不止 `vector<bool>`**：任何返回代理对象的场景都可能踩到同样的坑，比如某些数值库里用表达式模板（expression templates）优化矩阵/向量运算，中间表达式的返回类型也是一个只在整条语句内有效的代理对象，`auto` 直接接住这类代理类型再拖过语句边界使用同样危险。通用原则是：明确知道自己想要的目标类型时，用 `static_cast<目标类型>(表达式)` 去初始化 `auto` 变量，而不是让 `auto` 悄悄接住一个中间实现细节类型。

**要记住**
- "不可见"的代理类型会让 `auto` 推导出实现细节类型而非看起来应有的类型，这不是 `auto` 的错，而是代理类型本身在这类语境下容易被滥用。
- `std::vector<bool>::operator[]` 返回代理对象 `std::vector<bool>::reference` 是最经典的例子，用 `auto` 直接承接、并跨语句/跨临时对象生存期使用它是未定义行为的常见来源。
- 显式类型初始化惯用法：`auto var = static_cast<目标类型>(表达式);`，强制转换在初始化点立即发生。
- 这一惯用法同样适用于任何返回代理类型的 API（表达式模板等），不局限于 `vector<bool>`。

---

## 快速选择题

**1.**
```cpp
template<typename T>
void f(T& param) { }

int x = 27;
const int cx = x;
f(cx);
```
`T` 被推导为？
A. `int`
B. `const int`
C. `int&`
D. `const int&`

**答案：B** — `ParamType` 是 `T&`（非万能引用），规则是忽略实参的引用性但保留 const 性；`cx` 是 `const int`，所以 `T` 推导为 `const int`（`param` 类型为 `const int&`）。

---

**2.**
```cpp
template<typename T>
void f(T&& param) { }

int x = 27;
f(x);
f(27);
```
两次调用中 `T` 分别被推导为？
A. `int&`、`int`
B. `int`、`int&&`
C. `int&`、`int&&`
D. `int`、`int`

**答案：A** — `T&&` 是万能引用。传左值 `x` 时 `T` 推导为 `int&`（引用折叠后 `param` 为 `int&`）；传右值字面量 `27` 时按常规规则 `T` 推导为 `int`（`param` 为 `int&&`）。

---

**3.**
```cpp
template<typename T>
void f(T param) { }

const char name[] = "quant";
f(name);
```
`T` 被推导为？
A. `const char[6]`
B. `const char*`
C. `char*`
D. `const char&`

**答案：B** — 按值传递时数组名退化为指向首元素的指针，且顶层 const 在按值传递语境下与数组元素的 const 无关（这里数组元素是 `const char`，指针类型自然是 `const char*`）。

---

**4.**
```cpp
auto x1 = {1, 2, 3};

template<typename T>
void f(T param) { }
// f({1, 2, 3});
```
关于这两行，下列说法正确的是？
A. `x1` 推导为 `int`，`f({1,2,3})` 能编译
B. `x1` 推导为 `std::initializer_list<int>`，`f({1,2,3})` 编译失败
C. 两者都编译失败
D. `x1` 推导为 `std::vector<int>`，`f({1,2,3})` 能编译

**答案：B** — `auto` 遇到花括号初始化会特殊处理，推导为 `std::initializer_list<int>`；而普通模板类型推导无法从花括号列表推出 `T`，直接编译失败。

---

**5.**
```cpp
auto f() {
    return {1, 2, 3};
}
```
这段代码能否编译？
A. 能，`f` 返回 `std::initializer_list<int>`
B. 能，`f` 返回 `std::vector<int>`
C. 不能，函数返回类型位置的 `auto` 走模板类型推导，无法从花括号推出类型
D. 能，`f` 返回 `int`

**答案：C** — 函数返回类型的 `auto`（以及 C++14 泛型 lambda 的 `auto` 参数）使用模板类型推导规则而非"auto 变量"规则，花括号初始化在这个语境下无法被推导，编译失败。

---

**6.**
```cpp
int x = 0;
decltype(x) a = 1;
decltype((x)) b = a;
```
`a` 和 `b` 的类型分别是？
A. `int`、`int`
B. `int`、`int&`
C. `int&`、`int`
D. `int&`、`int&`

**答案：B** — `decltype(x)` 对裸变量名原样给出声明类型 `int`；`decltype((x))` 中 `(x)` 是括号包裹的左值表达式而非裸名字，规则切换为"该表达式类型的左值引用"，结果是 `int&`。

---

**7.**
```cpp
decltype(auto) f2() {
    int x = 0;
    return (x);
}
```
调用 `f2()` 会发生什么？
A. 正常返回 0
B. 编译错误
C. 返回一个悬空引用，未定义行为
D. `x` 被隐式转为静态变量，安全返回

**答案：C** — `return (x)` 中 `(x)` 触发 `decltype((x))` 规则，被推导为 `int&`，函数因此按引用返回一个即将销毁的局部变量，产生悬空引用，属于未定义行为。

---

**8.**
```cpp
template<typename Container, typename Index>
auto access(Container&& c, Index i) {
    return std::forward<Container>(c)[i];
}

std::vector<int> v{1, 2, 3};
access(v, 0) = 10;
```
这段代码能否编译？为什么？
A. 能，`access` 返回引用，可以被赋值
B. 不能，返回类型 `auto` 走模板类型推导按值返回，返回的是右值，不能作为赋值目标
C. 不能，`std::forward` 用法有语法错误
D. 能，因为 `v` 是左值容器

**答案：B** — 返回类型是普通 `auto`，走模板类型推导（按值传递规则），会把 `operator[]` 本应返回的引用类型剥离，`access(v, 0)` 返回一份拷贝，不能作为赋值的左值目标。若要保留引用性需要把返回类型改成 `decltype(auto)`。

---

**9.**
```cpp
template<typename T>
void f(const T& param) { }

int x = 0;
const int& rx = x;
f(rx);
std::cout << typeid(param).name();  // 假设在 f 内部输出
```
关于 `typeid(param).name()` 的输出，下列说法正确的是？
A. 一定精确打印出 `int const&`
B. 实现定义，且通常会丢失 const 和引用信息
C. 编译错误，`typeid` 不能用于引用
D. 输出结果和 `boost::typeindex` 完全一致

**答案：B** — `typeid(x).name()` 返回的字符串是实现定义的，多数实现会丢失 cv 限定符和引用性，只给出去掉这些修饰后的基础类型名；要精确保留这些信息需要 `boost::typeindex::type_id_with_cvr<T>().pretty_name()`。

---

**10.**
```cpp
std::vector<int> v(1'000'000, 1);
float sz = v.size();
```
这行代码最主要的问题是什么？
A. `v.size()` 返回类型和 `float` 无关，编译失败
B. `float` 精度有限（约 6-7 位十进制有效数字），存储足够大的 `size_type` 值会发生精度丢失
C. 没有问题，`float` 足以精确表示所有 `size_type` 值
D. `v.size()` 是运行期常量，`float` 只能存编译期常量

**答案：B** — `std::vector<int>::size_type` 是整型（通常与 `size_t` 一致，在 64 位平台上远超过 `float` 的有效精度），把它隐式转换/截断进 `float` 在数值足够大时会丢失精度，用 `auto sz = v.size();` 可以直接避免这个问题。

---

**11.**
```cpp
std::vector<bool> features(const Widget& w) { return {true, false, true}; }

Widget w;
auto highPriority = features(w)[2];
processPriority(highPriority);
```
这段代码的主要风险是什么？
A. `highPriority` 被推导为 `bool`，没有风险
B. `highPriority` 被推导为 `std::vector<bool>::reference` 代理对象，其指向的临时 vector 已被销毁，导致悬空引用
C. 编译错误，`vector<bool>` 不支持 `operator[]`
D. `highPriority` 是 `int`，存在隐式转换损耗

**答案：B** — `std::vector<bool>::operator[]` 返回代理类型 `std::vector<bool>::reference`（因为底层按 bit 压缩存储，无法返回真正的 `bool&`）。`features(w)` 返回临时对象，该语句结束后临时对象被销毁，而 `highPriority` 这个代理对象内部还持有指向该临时对象数据的指针，之后使用即为未定义行为。

---

**12.**
```cpp
auto highPriority = static_cast<bool>(features(w)[2]);
```
相比上一题的写法，这一行为什么能修复问题？
A. `static_cast` 让 `features(w)` 不再返回临时对象
B. `static_cast<bool>` 强制在临时 vector 销毁之前完成代理对象到 `bool` 的转换，`highPriority` 得到独立的 `bool` 值拷贝
C. `static_cast` 让 `auto` 不再进行类型推导
D. 这一行和上一题的写法本质相同，风险依然存在

**答案：B** — 这是"显式类型初始化惯用法"：用 `static_cast<目标类型>` 强制在初始化语句内、临时对象还存活时就完成代理类型到真实类型的转换，`auto` 因此推导出目标类型本身（这里是 `bool`）而不是依赖外部数据的代理对象，避免了悬空引用。
