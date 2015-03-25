#pragma once
#include <Snow/Cuda/CUDAVector.h>

/**
 * For the sake of supporting multiple implicit colliders in cuda, we define an enum for the type of collider
 * and use the ImplicitCollider.param to specify the collider once the type is known. Most simple implicit shapes
 * can be paramterized using at most 3 parameters. For instance, a half-plane is a point (ImplicitCollider.center)
 * and a normal (ImplicitCollider.param). A sphere is a center (ImplicitCollider.center) and a radius (ImplicitCollider.param.x)
 */

enum ColliderType
{
    HALF_PLANE = 0,
    SPHERE = 1
};

struct ImplicitCollider
{
	ColliderType type;
	CUDAVec3 center;
	CUDAVec3 param;
	CUDAVec3 velocity;
	float coeffFriction;

	__host__ __device__
		ImplicitCollider()
		: type(HALF_PLANE),
		center(0,0,0),
		param(0,1,0),
		velocity(0,0,0),
		coeffFriction(0.1f)
	{
	}

	__host__ __device__
		ImplicitCollider( ColliderType t, CUDAVec3 c, CUDAVec3 p = CUDAVec3(0,0,0), CUDAVec3 v = CUDAVec3(0,0,0), float f = 0.1f )
		: type(t),
		center(c),
		param(p),
		velocity(v),
		coeffFriction(f)
	{
		if ( p == CUDAVec3(0,0,0) ) {
			if ( t == HALF_PLANE ) p = CUDAVec3(0,1,0);
			else if ( t == SPHERE ) p = CUDAVec3(0.5f,0,0);
		}
	}

	__host__ __device__
		ImplicitCollider( const ImplicitCollider &collider )
		: type(collider.type),
		center(collider.center),
		param(collider.param),
		velocity(collider.velocity),
		coeffFriction(collider.coeffFriction)
	{
	}

	__host__ __device__
		void applyTransformation( const glm::mat4 &ctm )
	{
		glm::vec4 c = ctm * glm::vec4( glm::vec3(0,0,0), 1.f );
		center = CUDAVec3( c.x, c.y, c.z );
		switch ( type ) {
		case HALF_PLANE:
			{
				glm::vec4 n = ctm * glm::vec4( glm::vec3(0,1,0), 0.f );
				param = CUDAVec3( n.x, n.y, n.z );
				break;
			}
		case SPHERE:
			{
				const float *m = glm::value_ptr(ctm);
				param.x = sqrtf( m[0]*m[0] + m[1]*m[1] + m[2]*m[2] ); // Assumes uniform scale
				break;
			}
		}
	}
};