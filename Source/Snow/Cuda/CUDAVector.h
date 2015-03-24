#pragma once
#include <Utility/CUDAInclude.h>

struct CUDAVec3
{
    union 
	{
        float data[3];
        struct { float x, y, z; };
    };

    __host__ __device__ __forceinline__
    CUDAVec3() { x = 0.f; y = 0.f; z = 0.f; }

    __host__ __device__ __forceinline__
    CUDAVec3( float v ) { x = v; y = v; z = v; }

    __host__ __device__ __forceinline__
    CUDAVec3( float xx, float yy, float zz ) { x = xx; y = yy; z = zz; }

    __host__ __device__ __forceinline__
    CUDAVec3( const CUDAVec3 &v ) { x = v.x; y = v.y; z = v.z; }

    __host__ __device__ __forceinline__
    CUDAVec3( const glm::vec3 &v ) { x = v.x; y = v.y; z = v.z; }

    __host__ __device__ __forceinline__
    CUDAVec3( const glm::ivec3 &v ) { x = (float)v.x; y = (float)v.y; z = (float)v.z; }

    __host__ __device__ __forceinline__
    operator glm::vec3() const { return glm::vec3( x, y, z ); }

    __host__ __device__ __forceinline__
    operator glm::ivec3() const { return glm::ivec3( (int)x, (int)y, (int)z ); }

    __host__ __device__ __forceinline__
    CUDAVec3& operator = ( const CUDAVec3 &rhs ) { x = rhs.x; y = rhs.y; z = rhs.z; return *this; }

    __host__ __device__ __forceinline__
    CUDAVec3& operator = ( const glm::vec3 &rhs ) { x = rhs.x; y = rhs.y; z = rhs.z; return *this; }

    __host__ __device__ __forceinline__
    CUDAVec3& operator = ( const glm::ivec3 &rhs ) { x = (float)rhs.x; y = (float)rhs.y; z = (float)rhs.z; return *this; }

    __host__ __device__ __forceinline__
    int majorAxis() { return ( (fabsf(x)>fabsf(y)) ? ((fabsf(x)>fabsf(z)) ? 0 : 2) : ((fabsf(y)>fabsf(z)) ? 1 : 2) ); }

    __host__ __device__ __forceinline__
    float& operator [] ( int i ) { return data[i]; }

    __host__ __device__ __forceinline__
    float operator [] ( int i ) const { return data[i]; }

    __host__ __device__ __forceinline__
    static float dot( const CUDAVec3 &a, const CUDAVec3 &b ) { return a.x*b.x + a.y*b.y + a.z*b.z; }

    __host__ __device__ __forceinline__
    static CUDAVec3 cross( const CUDAVec3 &a, const CUDAVec3 &b )
    {
        return CUDAVec3( a.y*b.z - a.z*b.y,
                     a.z*b.x - a.x*b.z,
                     a.x*b.y - a.y*b.x );
    }

    __host__ __device__ __forceinline__
    static CUDAVec3 floor( const CUDAVec3 &v ) { return CUDAVec3( floorf(v.x), floorf(v.y), floorf(v.z) ); }

    __host__ __device__ __forceinline__
    static CUDAVec3 ceil( const CUDAVec3 &v ) { return CUDAVec3( ceilf(v.x), ceilf(v.y), ceilf(v.z) ); }

    __host__ __device__ __forceinline__
    static CUDAVec3 abs( const CUDAVec3 &v ) { return CUDAVec3( fabsf(v.x), fabsf(v.y), fabsf(v.z) ); }

    __host__ __device__ __forceinline__
    static CUDAVec3 round( const CUDAVec3 &v ) { return CUDAVec3( roundf(v.x), roundf(v.y), roundf(v.z) ); }

    //From http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
    __host__ __device__ __forceinline__
    static float sign( const float v ) { return (0.f < v) - (v < 0.f);}

    //From http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
    __host__ __device__ __forceinline__
    static CUDAVec3 sign( const CUDAVec3 &v ) { return CUDAVec3(sign(v.x), sign(v.y), sign(v.z) );}

    __host__ __device__ __forceinline__
    static CUDAVec3 min( const CUDAVec3 &v, const CUDAVec3 &w ) { return CUDAVec3( fminf(v.x, w.x), fminf(v.y, w.y), fminf(v.z,w.z) ); }

    __host__ __device__ __forceinline__
    static CUDAVec3 max( const CUDAVec3 &v, const CUDAVec3 &w ) { return CUDAVec3( fmaxf(v.x, w.x), fmaxf(v.y, w.y), fmaxf(v.z,w.z) ); }

    __host__ __device__ __forceinline__
    static CUDAVec3 mix(const CUDAVec3 &v, const CUDAVec3 &w, const CUDAVec3 &a) { return CUDAVec3(v.x*(1.f-a.x)+w.x*a.x, v.y*(1.f-a.y)+w.y*a.y, v.z*(1.f-a.z)+w.z*a.z); }

    __host__ __device__ __forceinline__
    static CUDAVec3 mix(const CUDAVec3 &v, const CUDAVec3 &w, float a) { return CUDAVec3(v.x*(1.f-a)+w.x*a, v.y*(1.f-a)+w.y*a, v.z*(1.f-a)+w.z*a); }

    __host__ __device__ __forceinline__
    static float length2( const CUDAVec3 &v ) { return v.x*v.x + v.y*v.y + v.z*v.z; }

    __host__ __device__ __forceinline__
    static float length( const CUDAVec3 &v ) { return sqrtf( v.x*v.x + v.y*v.y + v.z*v.z ); }

    __host__ __device__ __forceinline__
    static CUDAVec3 normalize( const CUDAVec3 &v ) { float f = 1.f/sqrtf(v.x*v.x+v.y*v.y+v.z*v.z); return CUDAVec3( f*v.x, f*v.y, f*v.z ); }

    __host__ __device__ __forceinline__
    CUDAVec3& mult (float f ) { x *= f; y *= f; z *= f; return *this;}

    __host__ __device__ __forceinline__
    CUDAVec3& add (float f ) { x += f; y += f; z += f; return *this;}

    __host__ __device__ __forceinline__
    CUDAVec3& add (const CUDAVec3 &v ) { x += v.x; y += v.y; z += v.z; return *this;}

    __host__ __device__ __forceinline__
    CUDAVec3& operator += ( const CUDAVec3 &rhs ) { x += rhs.x; y += rhs.y; z += rhs.z; return *this; }

    __host__ __device__ __forceinline__
    CUDAVec3 operator + ( const CUDAVec3 &rhs ) const { return CUDAVec3( x+rhs.x, y+rhs.y, z+rhs.z ); }

    __host__ __device__ __forceinline__
    CUDAVec3& operator -= ( const CUDAVec3 &rhs ) { x -= rhs.x; y -= rhs.y; z -= rhs.z; return *this; }

    __host__ __device__ __forceinline__
    CUDAVec3 operator - ( const CUDAVec3 &rhs ) const { return CUDAVec3( x-rhs.x, y-rhs.y, z-rhs.z ); }

    __host__ __device__ __forceinline__
    CUDAVec3& operator *= ( const CUDAVec3 &rhs ) { x *= rhs.x; y *= rhs.y; z *= rhs.z; return *this; }

    __host__ __device__ __forceinline__
    CUDAVec3 operator * ( const CUDAVec3 &rhs ) const { return CUDAVec3( x*rhs.x, y*rhs.y, z*rhs.z ); }

    __host__ __device__ __forceinline__
    CUDAVec3& operator /= ( const CUDAVec3 &rhs ) { x /= rhs.x; y /= rhs.y; z /= rhs.z; return *this; }

    __host__ __device__ __forceinline__
    CUDAVec3 operator / ( const CUDAVec3 &rhs ) const { return CUDAVec3( x/rhs.x, y/rhs.y, z/rhs.z ); }

    __host__ __device__ __forceinline__
    CUDAVec3& operator *= ( float f )  { x *= f; y *= f; z *= f; return *this; }

    __host__ __device__ __forceinline__
    CUDAVec3 operator * ( float f ) const { return CUDAVec3( f*x, f*y, f*z ); }

    __host__ __device__ __forceinline__
    CUDAVec3& operator *= ( double d )  { x = (float)(x*d); y = (float)(y*d); z = (float)(z*d); return *this; }

    __host__ __device__ __forceinline__
    CUDAVec3 operator * ( double d ) const { return CUDAVec3( (float)(x*d), (float)(y*d), (float)(z*d) ); }

    __host__ __device__ __forceinline__
    CUDAVec3& operator /= ( float f ) { float fi = 1.f/f; x *= fi; y *= fi; z *= fi; return *this; }

    __host__ __device__ __forceinline__
    CUDAVec3 operator / ( float f ) const { float fi = 1.f/f; return CUDAVec3( x*fi, y*fi, z*fi ); }

    __host__ __device__ __forceinline__
    CUDAVec3& operator += ( float f ) { x += f; y += f; z += f; return *this; }

    __host__ __device__ __forceinline__
    CUDAVec3 operator + ( float f ) const { return CUDAVec3( x+f, y+f, z+f ); }

    __host__ __device__ __forceinline__
    CUDAVec3& operator -= ( float f ) { x -= f; y -= f; z -= f; return *this; }

    __host__ __device__ __forceinline__
    CUDAVec3 operator - ( float f ) const { return CUDAVec3( x-f, y-f, z-f ); }

    __host__ __device__ __forceinline__
    bool valid( bool *nan = NULL ) const
    {
        if ( __isnanf(x) || __isnanf(y) || __isnanf(z) ) {
            if ( nan ) *nan = true;
            return false;
        } else if ( __isinff(x) || __isinff(y) || __isinff(z) ) {
            if ( nan ) *nan = false;
            return false;
        }
        return true;
    }

    __host__ __device__ __forceinline__
    static void print( const CUDAVec3 &v )
    {
        printf( "[%10f %10f %10f]\n", v.x, v.y, v.z );
    }

    __host__ __device__ __forceinline__
    bool operator == ( const CUDAVec3 &v ) const
    {
        return EQF( x, v.x ) && EQF( y, v.y ) && EQF( z, v.z );
    }

    __host__ __device__ __forceinline__
    bool operator != ( const CUDAVec3 &v ) const
    {
        return NEQF( x, v.x ) || NEQF( y, v.y ) || NEQF( z, v.z );
    }

};

__host__ __device__ __forceinline__
CUDAVec3 operator - ( const CUDAVec3 &v ) { return CUDAVec3( -v.x, -v.y, -v.z ); }

__host__ __device__ __forceinline__
CUDAVec3 operator * ( float f, const CUDAVec3 &v ) { return CUDAVec3( f*v.x, f*v.y, f*v.z ); }

__host__ __device__ __forceinline__
CUDAVec3 operator * ( double f, const CUDAVec3 &v ) { return CUDAVec3( (float)(f*v.x), (float)(f*v.y), (float)(f*v.z) ); }

