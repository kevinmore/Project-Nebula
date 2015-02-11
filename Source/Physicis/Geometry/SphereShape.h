#pragma once
#include "IShape.h"

class SphereShape : public IShape
{
public:

	/// Creates an arbitrary sphere with given center and radius.
	SphereShape()
		: IShape(IShape::SPHERE, vec3(0, 0, 0)),
		m_radius(0.5f)
	{}

	SphereShape(const vec3& center, float radius)
		: IShape(IShape::SPHERE, center),
		m_radius(radius)
	{}

	float getRadius() const { return m_radius; }
	void setRadius(float newRadius) { m_radius = newRadius; }

protected:
	float m_radius;
};



