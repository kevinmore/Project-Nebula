#include "SphereShape.h"


inline SphereShape::SphereShape()
	: AbstractShape(AbstractShape::SPHERE, vec3(0, 0, 0)),
	  m_radius(1.0f)
{
}

inline SphereShape::SphereShape(const vec3& center, float radius)
	: AbstractShape(AbstractShape::SPHERE, center),
	  m_radius(radius)
{
}

SphereShape::~SphereShape()
{
}

inline float SphereShape::getRadius() const
{
	return m_radius;
}


inline void SphereShape::setRadius( float newRadius )
{
	m_radius = newRadius;
}