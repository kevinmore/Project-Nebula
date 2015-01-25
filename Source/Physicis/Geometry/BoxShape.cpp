#include "BoxShape.h"

BoxShape::BoxShape()
	: AbstractShape(AbstractShape::BOX, vec3(0, 0, 0)),
	  m_halfExtents(vec3(0.5, 0.5, 0.5))
{
}

BoxShape::BoxShape( const vec3& center, const vec3& halfExtents )
	: AbstractShape(AbstractShape::BOX, center),
	  m_halfExtents(halfExtents)
{
}

vec3 BoxShape::getHalfExtents() const
{
	return m_halfExtents;
}

void BoxShape::setHalfExtents( const vec3& halfExtents )
{
	m_halfExtents = halfExtents;
}
