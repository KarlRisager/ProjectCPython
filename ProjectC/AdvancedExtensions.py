# -*- coding: utf-8 -*-
"""
@Project: LinalgDat2022
@File: AdvancedExtensions.py

@Description: Project C Determinant and Gram-Schmidt extensions.

"""

import math
import sys

sys.path.append('../Core')
from Vector import Vector
from Matrix import Matrix

Tolerance = 1e-6


def SquareSubMatrix(A: Matrix, i: int, j: int) -> Matrix:
    """
    This function creates the square submatrix given a square matrix as
    well as row and column indices to remove from it.

    Remarks:
        See page 246-247 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.

    Parameters:
        A:  N-by-N matrix
        i: int. The index of the row to remove.
        j: int. The index of the column to remove.

    Return:
        The resulting (N - 1)-by-(N - 1) submatrix.
    """
    N = A.M_Rows
    A1 = Matrix(N-1,N-1)
    rn = 0

    for r in range(N):
        if r!=i:
            cn = 0
            for c in range(N):
                if c!=j:
                    A1[r-rn,c-cn] = A[r,c]
                else:
                    cn+=1
                
        else:
            rn+=1
    return A1



    raise NotImplementedError()


def Determinant(A: Matrix) -> float:
    """
    This function computes the determinant of a given square matrix.

    Remarks:
        * See page 247 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.
        * Hint: Use SquareSubMatrix.

    Parameter:
        A: N-by-N matrix.

    Return:
        The determinant of the matrix.
    """
    N = A.M_Rows
    D = 0
    if A.Size == 1:
        return A[0,0]
    else:

        for j in range(N):

            D += (A[1,j]*(-1)**(1+j))*(Determinant(SquareSubMatrix(A,1,j)))
    return D

    raise NotImplementedError()


def VectorNorm(v: Vector) -> float:
    """
    This function computes the Euclidean norm of a Vector. This has been implemented
    in Project A and is provided here for convenience

    Parameter:
         v: Vector

    Return:
         Euclidean norm, i.e. (\sum v[i]^2)^0.5
    """
    nv = 0.0
    for i in range(len(v)):
        nv += v[i]**2
    return math.sqrt(nv)


def SetColumn(A: Matrix, v: Vector, j: int) -> Matrix:
    """
    This function copies Vector 'v' as a column of Matrix 'A'
    at column position j.

    Parameters:
        A: M-by-N Matrix.
        v: size M vector
        j: int. Column number.

    Return:
        Matrix A  after modification.

    Raise:
        ValueError if j is out of range or if len(v) != A.M_Rows.
    """
    M = A.M_Rows


    if len(v)!= M:
        raise ValueError()
    for i in range(M):
        A[i,j] = v[i]
    return A
    raise NotImplementedError()


def GramSchmidt(A: Matrix) -> tuple:
    """
    This function computes the Gram-Schmidt process on a given matrix.

    Remarks:
        See page 229 in "Linear Algebra for Engineers and Scientists"
        by K. Hardy.

    Parameter:
        A: M-by-N matrix. All columns are implicitly assumed linear
        independent.

    Return:
        tuple (Q,R) where Q is a M-by-N orthonormal matrix and R is an
        N-by-N upper triangular matrix.
    """
    M = A.M_Rows
    N = A.N_Cols
    Q = Matrix(M,N)
    R = Matrix(N,N)
    U = []
    for j in range(N):
        U.append(A.Column(j))


    q=U[0]
    for i in range(M):
        for j in range(N):

            if i<j:
                R[i,j] = Q.Column(i).__matmul__(U[j])

            elif i==j:
                R[i,j] = VectorNorm(q)
                q1 = q.__mul__((1/R[i,j]))
                SetColumn(Q,q1, j)

            elif i>j and i<N:
                R[i,j] = 0

        
        if i<N-1:
            q = U[i+1]
            for c in range(i+1):
                q1 = Q.Column(c).__rmul__(R[c,i+1])
                q = q.__sub__(q1)


    return (Q,R)
    raise NotImplementedError()