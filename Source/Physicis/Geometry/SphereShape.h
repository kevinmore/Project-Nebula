#pragma once
#include "AbstractShape.h"

class SphereShape : public AbstractShape
{
public:

	/// Creates an arbitrary sphere with given center and radius.
	inline SphereShape();
	inline SphereShape(const vec3& center, float radius);
	~SphereShape();

	inline float getRadius() const;
	inline void setRadius(float newRadius);

	

protected:
	float m_radius;
};



