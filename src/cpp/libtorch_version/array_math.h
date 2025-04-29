#pragma once
#include <cmath>
#include <array>

// Assign values from vector A to B, vector has dimension N
template<class T> inline void array_copy(T* B, const T* A, int N)
{
    for (int i = 0; i < N; i++) B[i] = A[i];
}

// Add elements of N-dimensional vectors A and B to get C[i] = A[i] + B[i]
template<class T> inline void array_add(T* C, const T* A, const T* B, int N)
{
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
}

// Subtract elements of N-dimensional vectors A and B to get C[i] = A[i] - B[i]
template<class T> inline void array_minus(T* C, const T* A, const T* B, int N)
{
    for (int i = 0; i < N; i++) C[i] = A[i] - B[i];
}

// Multiply each element of N-dimensional vector B by scalar A to get C[i] = B[i] * A
template<class T> inline void array_multi(T* C, const T* B, T A, int N)
{
    for (int i = 0; i < N; i++) C[i] = A * B[i];
}

// Compute the dot product of N-dimensional vectors A and B
template<class T> inline T array_dot(const T* A, const T* B, int N)
{
    T result = 0;
    for (int i = 0; i < N; i++) result += A[i] * B[i];
    return result;
}

// Compute the cross product C = A x B for 3-dimensional vectors; do not use V_Cross(A,A,B) or V_Cross(B,A,B)
template<class T> inline void array_cross(T* C, const T* A, const T* B)
{
    C[0] = A[1] * B[2] - A[2] * B[1];
    C[1] = A[2] * B[0] - A[0] * B[2];
    C[2] = A[0] * B[1] - A[1] * B[0];
}

// Compute the Euclidean (L2) norm of N-dimensional vector B
template<class T> inline T array_norm2(const T* B, int N)
{
    T result = array_dot(B, B, N);
    return std::sqrt(result);
}




using Vector_RorV = std::array<double, 3>;
using Matrix_Rotate = std::array<Vector_RorV, 3>;

// Compute the magnitude (norm) of a 3D vector
inline double norm(const Vector_RorV& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// Compute the dot product of two 3D vectors
inline double dot(const Vector_RorV& a, const Vector_RorV& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Compute the cross product of two 3D vectors
inline Vector_RorV cross(const Vector_RorV& a, const Vector_RorV& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

// Normalize a 3D vector to a unit vector
inline Vector_RorV unitVector(const Vector_RorV& v) {
    double n = norm(v);
    return { v[0] / n, v[1] / n, v[2] / n };
}

// Build a rotation matrix that rotates vector 'a' to vector 'b'
inline Matrix_Rotate rotationMatrix(const Vector_RorV& a, const Vector_RorV& b) {
    Vector_RorV u = unitVector(a);
    Vector_RorV v = unitVector(b);
    double theta = std::acos(dot(u, v));
    // If the angle is near zero, return identity matrix
    if (theta < 1e-6) {
        return { {
            { 1, 0, 0 },
            { 0, 1, 0 },
            { 0, 0, 1 }
        } };
    }
    // Compute rotation axis as unit cross product of u and v
    Vector_RorV w = unitVector(cross(u, v));
    double ct = std::cos(theta);
    double st = std::sin(theta);
    // Rodrigues' rotation formula
    Matrix_Rotate R = { {
        { ct + w[0] * w[0] * (1 - ct),    w[0] * w[1] * (1 - ct) - w[2] * st,  w[0] * w[2] * (1 - ct) + w[1] * st },
        { w[1] * w[0] * (1 - ct) + w[2] * st,ct + w[1] * w[1] * (1 - ct),      w[1] * w[2] * (1 - ct) - w[0] * st },
        { w[2] * w[0] * (1 - ct) - w[1] * st,w[2] * w[1] * (1 - ct) + w[0] * st,  ct + w[2] * w[2] * (1 - ct) }
    } };
    return R;
}

// Multiply a 3x3 matrix by a 3D vector
inline Vector_RorV multiply(const Matrix_Rotate& M, const Vector_RorV& v) {
    return {
        M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
        M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
        M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]
    };
}

// Build a rotation matrix for a rotation around the X-axis by 'theta' radians
inline Matrix_Rotate rotateX(double theta) {
    double ct = std::cos(theta);
    double st = std::sin(theta);
    return { {
        { 1,  0,   0 },
        { 0,  ct, -st },
        { 0,  st,  ct }
    } };
}

// Rotate two position-velocity vectors (rv0 and rv1) into a common frame
inline Matrix_Rotate rv_rotate(double* rv0, double* rv1,
    double* rv0_rotated, double* rv1_rotated) {
    // Extract position P and velocity V from rv0
    Vector_RorV P = { rv0[0], rv0[1], rv0[2] };
    Vector_RorV V = { rv0[3], rv0[4], rv0[5] };

    // Compute normal vector n = rv0 x rv1
    double n_vector[3];
    array_cross(n_vector, rv0, rv1);

    // If normal is non-zero, decompose velocity into normal and tangential components
    if (array_norm2(n_vector, 3) > 0.0) {
        double n = array_norm2(n_vector, 3);
        array_multi(n_vector, n_vector, 1.0 / n, 3);

        double vdotp = array_dot(rv0 + 3, n_vector, 3);
        double v_n[3];
        array_multi(v_n, n_vector, vdotp, 3);
        double v_t[3];
        array_minus(v_t, rv0 + 3, v_n, 3);

        V = { v_t[0], v_t[1], v_t[2] };
    }

    // Extract P1 and V1 from rv1
    Vector_RorV P1 = { rv1[0], rv1[1], rv1[2] };
    Vector_RorV V1 = { rv1[3], rv1[4], rv1[5] };

    // Define target direction as positive X-axis
    Vector_RorV target = { 1, 0, 0 };

    // Compute first rotation R1 to align P with the X-axis
    Matrix_Rotate R1 = rotationMatrix(P, target);

    // Apply R1 to P and V
    Vector_RorV P_rot = multiply(R1, P);
    Vector_RorV V_rot = multiply(R1, V);
    Vector_RorV P1_rot = multiply(R1, P1);
    Vector_RorV V1_rot = multiply(R1, V1);

    // Compute angle phi to rotate around X-axis so that V lies in the XY-plane
    double phi = -std::atan2(V_rot[2], V_rot[1]);
    Matrix_Rotate Rx = rotateX(phi);

    // Compute total rotation R_total = Rx * R1
    Matrix_Rotate R_total;
    // Matrix multiplication Rx * R1
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_total[i][j] = Rx[i][0] * R1[0][j] +
                Rx[i][1] * R1[1][j] +
                Rx[i][2] * R1[2][j];
        }
    }

    // Restore original velocity before final rotation
    V = { rv0[3], rv0[4], rv0[5] };

    // Apply final rotation to both vectors
    P_rot = multiply(R_total, P);
    V_rot = multiply(R_total, V);
    P1_rot = multiply(R_total, P1);
    V1_rot = multiply(R_total, V1);

    // Store rotated values back into output arrays
    for (int i = 0; i < 3; i++) {
        rv0_rotated[i] = P_rot[i];
        rv1_rotated[i] = P1_rot[i];
        rv0_rotated[i + 3] = V_rot[i];
        rv1_rotated[i + 3] = V1_rot[i];
    }

    return R_total;
}