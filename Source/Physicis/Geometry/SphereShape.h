#pragma once
#include "AbstractShape.h"

class SphereShape : public AbstractShape
{
public:

	/// Creates an arbitrary sphere with given center and radius.
	SphereShape()
		: AbstractShape(AbstractShape::SPHERE, vec3(0, 0, 0)),
		m_radius(0.5f)
	{}

	SphereShape(const vec3& center, float radius)
		: AbstractShape(AbstractShape::SPHERE, center),
		m_radius(radius)
	{}

	float getRadius() const { return m_radius; }
	void setRadius(float newRadius) { m_radius = newRadius; }

protected:
	float m_radius;
};



