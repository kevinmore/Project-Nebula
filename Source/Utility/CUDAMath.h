#pragma once
#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifdef EPSILON
#undef EPSILON
#endif

#ifdef _EPSILON_
#undef _EPSILON_
#endif

#define _EPSILON_ 1e-6
#define EPSILON _EPSILON_

#define EQ(a, b) ( fabs((a) - (b)) < _EPSILON_ )
#define NEQ(a, b) ( fabs((a) - (b)) > _EPSILON_ )

#define EQF(a, b) ( fabsf((a) - (b)) < _EPSILON_ )
#define NEQF(a, b) ( fabsf((a) - (b)) > _EPSILON_ )

#ifdef MIN
#undef MIN
#endif
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))

#ifdef MAX
#undef MAX
#endif
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))

#ifdef CLAMP
#undef CLAMP
#endif
#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

static inline float urand( float min = 0.f, float max = 1.f )
{
	return min + (float(rand())/float(RAND_MAX))*(max-min);
}

static inline float smoothstep( float value, float edge0, float edge1 )
{
	float x = CLAMP( (value-edge0)/(edge1-edge0), 0.f, 1.f );
	return x*x*(3-2*x);
}

static inline float smootherstep( float value, float edge0, float edge1 )
{
	float x = CLAMP( (value-edge0)/(edge1-edge0), 0.f, 1.f );
	return x*x*x*(x*(x*6-15)+10);
}