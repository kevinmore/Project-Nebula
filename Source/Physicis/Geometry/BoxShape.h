#pragma once
#include "IShape.h"

class BoxShape : public IShape
{
public:
	/// Creates a box with the given half extents ( An (X by Y by Z) box has the half-extents (X/2, Y/2, Z/2) ).
	BoxShape()
		: IShape(IShape::BOX, vec3(0, 0, 0)),
		m_halfExtents(vec3(0.5, 0.5, 0.5))
	{}

	BoxShape(const vec3& center, const vec3& halfExtents)
		: IShape(IShape::BOX, center),
		m_halfExtents(halfExtents)
	{}

	vec3 getHalfExtents() const { return m_halfExtents; }
	void setHalfExtents(const vec3& halfExtents) { m_halfExtents = halfExtents; }

protected:
	vec3 m_halfExtents;
};

