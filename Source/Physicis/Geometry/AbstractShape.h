#pragma once
#include <Utility/EngineCommon.h>

class AbstractShape
{
public:

	enum ShapeType
	{
		SPHERE,			
		CYLINDER,
		TRIANGLE,		
		BOX,		
		CAPSULE,
		CONVEXBODY
	};

	AbstractShape(ShapeType shapeType, const vec3& pt) 
	{ 
		m_type = shapeType; 
		m_pos = pt;
	}
	~AbstractShape() {}

	inline const vec3& getPosition() const { return m_pos; } 
	inline void setPosition(const vec3& newPos) { m_pos = newPos; }

	ShapeType m_type;

protected:
	vec3 m_pos;
};

