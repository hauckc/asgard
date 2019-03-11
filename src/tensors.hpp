#pragma once

#include <string>
#include <vector>

/* tolerance for answer comparisons */
#define TOL std::numeric_limits<P>::epsilon() * 2

enum class mem_type
{
  owner,
  view
};

template<mem_type mem>
using enable_for_owner = std::enable_if_t<mem == mem_type::owner>;

template<mem_type mem>
using enable_for_view = std::enable_if_t<mem == mem_type::view>;

namespace fk
{
// forward declarations
template<typename P, mem_type mem = mem_type::owner> // default to be an owner
class vector;
template<typename P>
class matrix;

template<typename P, mem_type mem>
class vector
{
public:
  template<mem_type m_ = mem, typename = enable_for_owner<m_>>
  vector();
  explicit vector(int const size);
  vector(std::initializer_list<P> list);
  vector(std::vector<P> const &);
  vector(fk::matrix<P> const &);
  ~vector();

  // same to same precision copy constructor/assignment
  template<mem_type omem>
  vector(vector<P, omem> const &);
  template<mem_type omem>
  vector<P, mem> &operator=(vector<P, omem> const &);

  // converting precision copy constructor/assignment
  template<typename PP, mem_type omem>
  vector(vector<PP, omem> const &);
  template<typename PP, mem_type omem>
  vector<P, mem> &operator=(vector<PP, omem> const &);

  // same to same move precision constructor/assignment
  template<mem_type omem>
  vector(vector<P, omem> &&);
  template<mem_type omem>
  vector<P, mem> &operator=(vector<P, omem> &&);

  //
  // copy out of std::vector
  //
  vector<P, mem> &operator=(std::vector<P> const &);

  //
  // copy into std::vector
  //
  std::vector<P> to_std() const;

  //
  // subscripting operators
  //
  P &operator()(int const);
  P operator()(int const) const;
  //
  // comparison operators
  //
  bool operator==(vector<P, mem> const &) const;
  bool operator!=(vector<P, mem> const &) const;
  bool operator<(vector<P, mem> const &) const;
  //
  // math operators
  //
  vector<P, mem> operator+(vector<P, mem> const &right) const;
  vector<P, mem> operator-(vector<P, mem> const &right) const;
  P operator*(vector<P, mem> const &)const;
  vector<P, mem> operator*(matrix<P> const &)const;
  vector<P, mem> operator*(P const) const;
  vector<P, mem> single_column_kron(vector<P, mem> const &) const;

  //
  // basic queries to private data
  //
  int size() const { return size_; }
  // just get a pointer. cannot deref/assign. for e.g. blas
  // use subscript operators for general purpose access
  P *data(int const elem = 0) const { return &data_[elem]; }
  //
  // utility functions
  //
  void print(std::string const label = "") const;
  void dump_to_octave(char const *) const;
  void resize(int const size = 0);
  vector<P, mem> &set(int const, vector<P, mem> const);
  vector<P, mem> extract(int const, int const) const;
  vector<P, mem> &concat(vector<P, mem> const &right);
  typedef P *iterator;
  typedef const P *const_iterator;
  iterator begin() { return data(); }
  iterator end() { return data() + size(); }
  const_iterator begin() const { return data(); }
  const_iterator end() const { return data() + size(); }

private:
  P *data_;  //< pointer to elements
  int size_; //< dimension
};

template<typename P>
class matrix
{
public:
  matrix();
  matrix(int rows, int cols);
  matrix(std::initializer_list<std::initializer_list<P>> list);

  ~matrix();

  matrix(matrix<P> const &);
  matrix<P> &operator=(matrix<P> const &);
  template<typename PP>
  matrix(matrix<PP> const &);
  template<typename PP>
  matrix<P> &operator=(matrix<PP> const &);

  matrix(matrix<P> &&);
  matrix<P> &operator=(matrix<P> &&);

  //
  // copy out of fk::vector
  //
  matrix<P> &operator=(fk::vector<P> const &);
  //
  // subscripting operators
  //
  P &operator()(int const, int const);
  P operator()(int const, int const) const;
  //
  // comparison operators
  //
  bool operator==(matrix<P> const &) const;
  bool operator!=(matrix<P> const &) const;
  bool operator<(matrix<P> const &) const;
  //
  // math operators
  //
  matrix<P> operator*(P const) const;
  vector<P> operator*(vector<P> const &)const;
  matrix<P> operator*(matrix<P> const &)const;
  matrix<P> operator+(matrix<P> const &) const;
  matrix<P> operator-(matrix<P> const &) const;

  matrix<P> &transpose();

  matrix<P> kron(matrix<P> const &) const;

  // clang-format off
  template<typename U = P>
  std::enable_if_t<
    std::is_floating_point<U>::value && std::is_same<P, U>::value,
  matrix<P> &> invert();


  template<typename U = P>
  std::enable_if_t<
      std::is_floating_point<U>::value && std::is_same<P, U>::value,
  P> determinant() const;
  // clang-format on

  //
  // basic queries to private data
  //
  int nrows() const { return nrows_; }
  int ncols() const { return ncols_; }
  int size() const { return nrows() * ncols(); }
  // just get a pointer. cannot deref/assign. for e.g. blas
  // use subscript operators for general purpose access
  P *data(int const i = 0, int const j = 0) const
  {
    // return &data_[i * ncols() + j]; // row-major
    return &data_[j * nrows() + i]; // column-major
  }
  //
  // utility functions
  //

  matrix<P> &update_col(int const, fk::vector<P> const &);
  matrix<P> &update_col(int const, std::vector<P> const &);
  matrix<P> &update_row(int const, fk::vector<P> const &);
  matrix<P> &update_row(int const, std::vector<P> const &);
  matrix<P> &clear_and_resize(int const, int const);
  matrix<P> &set_submatrix(int const row_idx, int const col_idx,
                           fk::matrix<P> const &submatrix);
  matrix<P> extract_submatrix(int const row_idx, int const col_idx,
                              int const num_rows, int const num_cols) const;
  void print(std::string const label = "") const;
  void dump_to_octave(char const *name) const;

  typedef P *iterator;
  typedef const P *const_iterator;
  iterator begin() { return data(); }
  iterator end() { return data() + size(); }
  const_iterator begin() const { return data(); }
  const_iterator end() const { return data() + size(); }

private:
  P *data_;   //< pointer to elements
  int nrows_; //< row dimension
  int ncols_; //< column dimension
};

} // namespace fk

// suppress implicit instantiations later on
// implies fk::vector<double, mem_type::owner>
extern template class fk::vector<double>;
extern template class fk::vector<float>;
extern template class fk::vector<int>;
extern template class fk::matrix<double>;
extern template class fk::matrix<float>;
extern template class fk::matrix<int>;

extern template fk::vector<int>::vector(vector<float> const &);
extern template fk::vector<int>::vector(vector<double> const &);
extern template fk::vector<float>::vector(vector<int> const &);
extern template fk::vector<float>::vector(vector<double> const &);
extern template fk::vector<double>::vector(vector<int> const &);
extern template fk::vector<double>::vector(vector<float> const &);

extern template fk::vector<int> &fk::vector<int>::
operator=(vector<float> const &);
extern template fk::vector<int> &fk::vector<int>::
operator=(vector<double> const &);
extern template fk::vector<float> &fk::vector<float>::
operator=(vector<int> const &);
extern template fk::vector<float> &fk::vector<float>::
operator=(vector<double> const &);
extern template fk::vector<double> &fk::vector<double>::
operator=(vector<int> const &);
extern template fk::vector<double> &fk::vector<double>::
operator=(vector<float> const &);

extern template fk::matrix<int>::matrix(matrix<float> const &);
extern template fk::matrix<int>::matrix(matrix<double> const &);
extern template fk::matrix<float>::matrix(matrix<int> const &);
extern template fk::matrix<float>::matrix(matrix<double> const &);
extern template fk::matrix<double>::matrix(matrix<int> const &);
extern template fk::matrix<double>::matrix(matrix<float> const &);

extern template fk::matrix<int> &fk::matrix<int>::
operator=(matrix<float> const &);
extern template fk::matrix<int> &fk::matrix<int>::
operator=(matrix<double> const &);
extern template fk::matrix<float> &fk::matrix<float>::
operator=(matrix<int> const &);
extern template fk::matrix<float> &fk::matrix<float>::
operator=(matrix<double> const &);
extern template fk::matrix<double> &fk::matrix<double>::
operator=(matrix<int> const &);
extern template fk::matrix<double> &fk::matrix<double>::
operator=(matrix<float> const &);

// remove these when matrix::invert()/determinatn() is availble for ints
extern template fk::matrix<float> &fk::matrix<float>::invert();
extern template fk::matrix<double> &fk::matrix<double>::invert();
extern template float fk::matrix<float>::determinant() const;
extern template double fk::matrix<double>::determinant() const;

// added for mem_type support

extern template fk::vector<int, mem_type::owner>::vector(); // needed b/c of
                                                            // sfinae decl
extern template fk::vector<float, mem_type::owner>::vector();
extern template fk::vector<double, mem_type::owner>::vector();

extern template class fk::vector<double, mem_type::view>; // get the non-default
                                                          // mem_type::view
extern template class fk::vector<float, mem_type::view>;
extern template class fk::vector<int, mem_type::view>;

extern template fk::vector<double, mem_type::owner>::vector(
    vector<double, mem_type::owner> &&);
extern template fk::vector<float, mem_type::owner>::vector(
    vector<float, mem_type::owner> &&);
extern template fk::vector<int, mem_type::owner>::vector(
    vector<int, mem_type::owner> &&);
extern template fk::vector<double, mem_type::owner> &
fk::vector<double, mem_type::owner>::
operator=(vector<double, mem_type::owner> &&);
extern template fk::vector<float, mem_type::owner> &
fk::vector<float, mem_type::owner>::
operator=(vector<float, mem_type::owner> &&);
extern template fk::vector<int, mem_type::owner> &
fk::vector<int, mem_type::owner>::operator=(vector<int, mem_type::owner> &&);
