#pragma once
#include "IShape.h"
#include <Snow/Grid.h>
#include <Utility/Math.h>

class BoxShape : public IShape
{
public:
	/// Creates a box with the given half extents ( An (X by Y by Z) box has the half-extents (X/2, Y/2, Z/2) ).
	BoxShape()
		: IShape(IShape::BOX, vec3(0, 0, 0)),
		m_halfExtents(vec3(0.5, 0.5, 0.5)),
		m_min(m_center - m_halfExtents),
		m_max(m_center + m_halfExtents)
	{}

	BoxShape(const vec3& center, const vec3& halfExtents)
		: IShape(IShape::BOX, center),
		m_halfExtents(halfExtents),
		m_min(m_center - m_halfExtents),
		m_max(m_center + m_halfExtents)
	{}

	BoxShape( const BoxShape &other ) 
		: IShape(IShape::BOX, other.getCenter()),
		  m_min(other.m_min), 
		  m_max(other.m_max), 
		  m_halfExtents(other.getHalfExtents()) 
	{}

	vec3 getHalfExtents() const { return m_halfExtents; }
	inline void setHalfExtents(const vec3& halfExtents) 
	{ 
		m_halfExtents = halfExtents; 
		m_min = m_center - m_halfExtents;
		m_max = m_center + m_halfExtents;
	}

	vec3 getMin() const { return m_min; }
	vec3 getMax() const { return m_max; }

	/// Convert the bounding box to a grid for snow simulation
	inline Grid toGrid(float h) const
	{
		BoxShape box(*this);
		box.expandAbs( h );
		box.fix( h );
		Grid grid;
		CUDAVec3 bmax = Math::Converter::toCUDAVec3(box.m_max);
		CUDAVec3 bmin = Math::Converter::toCUDAVec3(box.m_min);
		CUDAVec3 dimf = CUDAVec3::round( (bmax-bmin)/h );
		grid.dim = glm::ivec3( dimf.x, dimf.y, dimf.z );
		grid.h = h;
		grid.pos = bmin;
		return grid;
	}

	// Expand box by absolute distances
	inline void expandAbs( float d ) { m_min -= vec3(d, d, d); m_max += vec3(d, d, d); }
	inline void expandAbs( const vec3 &d ) { m_min -= d; m_max += d; }

	// Expand box relative to current size
	inline void expandRel( float d ) { vec3 dd = d*(m_max-m_min); m_min -= dd; m_max += dd; }
	inline void expandRel( const vec3 &d ) { vec3 dd = d*(m_max-m_min); m_min -= dd; m_max += dd; }

	inline void fix( float h )
	{
		float ceilX = ceilf((m_max - m_min).x()/h);
		float ceilY = ceilf((m_max - m_min).y()/h);
		float ceilZ = ceilf((m_max - m_min).z()/h);
		vec3 d = 0.5f * h * vec3(ceilX, ceilY, ceilZ);

		m_min = m_center - d;
		m_max = m_center + d;
	}

protected:
	vec3 m_halfExtents;
	vec3 m_min;
	vec3 m_max;
};

