#pragma once
#include <Utility/EngineCommon.h>

class IShape
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

	IShape(ShapeType shapeType, const vec3& pt) 
	{ 
		m_type = shapeType; 
		m_center = pt;
	}
	~IShape() {}

	inline const vec3& getCenter() const { return m_center; } 
	inline void setCenter(const vec3& newCenter) { m_center = newCenter; }

	ShapeType m_type;

protected:
	vec3 m_center;
};

