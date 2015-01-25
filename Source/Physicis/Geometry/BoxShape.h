#pragma once
#include "AbstractShape.h"

class BoxShape : public AbstractShape
{
public:
	/// Creates a box with the given half extents ( An (X by Y by Z) box has the half-extents (X/2, Y/2, Z/2) ).
	BoxShape();
	BoxShape(const vec3& center, const vec3& halfExtents);

	vec3 getHalfExtents() const;
	void setHalfExtents(const vec3& halfExtents);

protected:
	vec3 m_halfExtents;
};

