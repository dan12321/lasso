use lasso::linalg::{Matrix};

#[test]
fn get_new() {
    let matrix: Matrix<usize> = Matrix::new(2, 3);
    assert_eq!(matrix.get(0, 0).unwrap(), 0);
    assert_eq!(matrix.get(1, 2).unwrap(), 0);
}

#[test]
fn get_new_value() {
    let matrix: Matrix<usize> = Matrix::new_fill(2, 3, 4);
    assert_eq!(matrix.get(0, 0).unwrap(), 4);
    assert_eq!(matrix.get(1, 2).unwrap(), 4);
}

#[test]
fn get_new_error() {
    let matrix: Matrix<usize> = Matrix::new(2, 3);
    assert!(matrix.get(2, 0).is_err());
    assert!(matrix.get(0, 3).is_err());
}

#[test]
fn set() {
    let mut matrix: Matrix<usize> = Matrix::new(2, 3);
    let result = matrix.set(1, 2, 4);
    assert!(result.is_ok());
    assert_eq!(matrix.get(1, 2).unwrap(), 4);
    assert_eq!(matrix.get(0, 0).unwrap(), 0);
}

#[test]
fn set_error() {
    let mut matrix: Matrix<usize> = Matrix::new(2, 3);
    assert!(matrix.set(2, 0, 4).is_err());
    assert!(matrix.set(0, 3, 4).is_err());
}

#[test]
fn scalar_multiplication() {
    let mut matrix: Matrix<usize> = Matrix::new(2, 2);
    matrix.set(1, 0, 1).unwrap();
    matrix.set(0, 1, 2).unwrap();
    matrix.set(1, 1, 3).unwrap();
    let result = matrix.mul_scalar(5);
    assert_eq!(result.get(0, 0).unwrap(), 0);
    assert_eq!(result.get(1, 0).unwrap(), 5);
    assert_eq!(result.get(0, 1).unwrap(), 10);
}

#[test]
fn matrix_addition() {
    let mut matrix: Matrix<usize> = Matrix::new(2, 3);
    matrix.set(1, 0, 1).unwrap();
    matrix.set(0, 1, 2).unwrap();
    matrix.set(1, 1, 3).unwrap();
    let mut second_matrix: Matrix<usize> = Matrix::new(2, 3);
    second_matrix.set(1, 0, 2).unwrap();
    second_matrix.set(0, 1, 3).unwrap();
    second_matrix.set(1, 1, 4).unwrap();
    let result = matrix.add(&second_matrix).unwrap();
    assert_eq!(result.get(0, 0).unwrap(), 0);
    assert_eq!(result.get(1, 0).unwrap(), 3);
    assert_eq!(result.get(0, 1).unwrap(), 5);
    assert_eq!(result.get(1, 1).unwrap(), 7);
}

#[test]
fn matrix_addition_wrong_sizes() {
    let mut matrix: Matrix<usize> = Matrix::new(2, 3);
    matrix.set(1, 0, 1).unwrap();
    matrix.set(0, 1, 2).unwrap();
    matrix.set(1, 1, 3).unwrap();
    let mut second_matrix: Matrix<usize> = Matrix::new(2, 4);
    second_matrix.set(1, 0, 2).unwrap();
    second_matrix.set(0, 1, 3).unwrap();
    second_matrix.set(1, 1, 4).unwrap();
    assert!(matrix.add(&second_matrix).is_err());
}

#[test]
fn matrix_multiplication() {
    let mut matrix: Matrix<usize> = Matrix::new(2, 3);
    matrix.set(1, 0, 1).unwrap();
    matrix.set(0, 1, 2).unwrap();
    matrix.set(1, 1, 3).unwrap();
    matrix.set(0, 2, 4).unwrap();
    matrix.set(1, 2, 5).unwrap();
    let mut second_matrix: Matrix<usize> = Matrix::new(3, 2);
    second_matrix.set(1, 0, 2).unwrap();
    second_matrix.set(2, 0, 3).unwrap();
    second_matrix.set(0, 1, 4).unwrap();
    second_matrix.set(1, 1, 5).unwrap();
    second_matrix.set(2, 1, 6).unwrap();
    let result = matrix.mul(&second_matrix).unwrap();
    assert_eq!(result.get(0, 0).unwrap(), 4);
    assert_eq!(result.get(1, 0).unwrap(), 5);
    assert_eq!(result.get(2, 0).unwrap(), 6);
    assert_eq!(result.get(0, 1).unwrap(), 12);
    assert_eq!(result.get(1, 1).unwrap(), 19);
    assert_eq!(result.get(2, 1).unwrap(), 24);
    assert_eq!(result.get(0, 2).unwrap(), 20);
    assert_eq!(result.get(1, 2).unwrap(), 33);
    assert_eq!(result.get(2, 2).unwrap(), 42);
    assert!(result.get(3, 0).is_err());
    assert!(result.get(0, 3).is_err());
}

#[test]
fn matrix_multiplication_wrong_sizes() {
    let mut matrix: Matrix<usize> = Matrix::new(2, 3);
    matrix.set(1, 0, 1).unwrap();
    matrix.set(0, 1, 2).unwrap();
    matrix.set(1, 1, 3).unwrap();
    let mut second_matrix: Matrix<usize> = Matrix::new(2, 2);
    second_matrix.set(1, 0, 2).unwrap();
    second_matrix.set(0, 1, 3).unwrap();
    second_matrix.set(1, 1, 4).unwrap();
    assert!(matrix.mul(&second_matrix).is_ok());
    assert!(second_matrix.mul(&matrix).is_err());
}

#[test]
fn dot_product() {
    let mut vector: Matrix<usize> = Matrix::new(3, 1);
    vector.set(0, 0, 0).unwrap();
    vector.set(1, 0, 1).unwrap();
    vector.set(2, 0, 2).unwrap();
    let mut second_vector: Matrix<usize> = Matrix::new(3, 1);
    second_vector.set(0, 0, 3).unwrap();
    second_vector.set(1, 0, 4).unwrap();
    second_vector.set(2, 0, 5).unwrap();
    let result = vector.dot(&second_vector).unwrap();
    assert_eq!(result, 14);
}

#[test]
fn dot_product_wrong_sizes() {
    let mut vector: Matrix<usize> = Matrix::new(3, 1);
    vector.set(0, 0, 0).unwrap();
    vector.set(1, 0, 1).unwrap();
    vector.set(2, 0, 2).unwrap();
    let mut second_vector: Matrix<usize> = Matrix::new(4, 1);
    second_vector.set(0, 0, 3).unwrap();
    second_vector.set(1, 0, 4).unwrap();
    second_vector.set(2, 0, 5).unwrap();
    let result = vector.dot(&second_vector);
    assert!(result.is_err());
}

#[test]
fn get_row() {
    let mut matrix: Matrix<usize> = Matrix::new(2, 3);
    matrix.set(1, 0, 1).unwrap();
    matrix.set(0, 1, 2).unwrap();
    matrix.set(1, 1, 3).unwrap();
    let vector = matrix.row(1).unwrap();
    assert_eq!(vector.get(0, 0).unwrap(), 2);
    assert_eq!(vector.get(0, 1).unwrap(), 3);
    assert!(vector.get(1, 0).is_err());
    assert!(vector.get(0, 2).is_err());
}

#[test]
fn get_row_out_of_bounds() {
    let mut matrix: Matrix<usize> = Matrix::new(2, 3);
    matrix.set(1, 0, 1).unwrap();
    matrix.set(0, 1, 2).unwrap();
    matrix.set(1, 1, 3).unwrap();
    let result = matrix.row(3);
    assert!(result.is_err());
}

#[test]
fn new_random_fill() {
    let matrix: Matrix<f64> = Matrix::new_random_fill(1000, 1000, 0.0, 1.0);
    let element = matrix.get(0, 0).unwrap();
    let mut random = false;
    for i in 0..matrix.width() {
        for j in 0..matrix.height() {
            if element != matrix.get(i, j).unwrap() {
                random = true;
                break;
            }
        }
        if random {
            break;
        }
    }
    assert!(random);
}