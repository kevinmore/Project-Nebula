#pragma once
#include <Utility/Math.h>
using namespace Math;
class Transform

{
public:
	Transform();
	Transform(const Transform& other);
	Transform(const vec3& translation, const quat& rotation, const vec3& scale = Vector3::UNIT_SCALE);

	/// Static utility functions
	static vec3 computeLocalPosition(const vec3& worldPos, const Transform& transform);

	/// Position Access
	inline const vec3& getPosition() const { return m_position; }
	inline void setPosition(const vec3& pos) { m_position = pos; } 
	inline void setPosition(float x, float y, float z) { m_position = vec3(x, y, z); }
	inline void setPositionX(float x) { m_position.setX(x); }
	inline void setPositionY(float y) { m_position.setY(y); }
	inline void setPositionZ(float z) { m_position.setZ(z); }
	inline void translate(const vec3& delta) { m_position += delta; }

	/// Rotation Access
	inline const quat& getRotation() const { return m_rotation; }
	inline const vec3& getEulerAngles() const { return m_eulerAngles; }
	inline glm::mat3 getRotationMatrix() const
	{
		return glm::mat3_cast(Converter::toGLMQuat(m_rotation));
	}

	// sets the rotation with a given quaternion
	// this will not update the euler angles
	inline void setRotation(const quat& rot) 
	{ 
		m_rotation = rot;
// 		glm::vec3 eulerAnglesInRadians = glm::eulerAngles(Converter::toGLMQuat(m_rotation));
// 		m_eulerAngles = Converter::toQtVec3(glm::degrees(eulerAnglesInRadians));
	}

	// sets the rotation with a given euler angles in degrees
	inline void setRotation(const vec3& eulerAngles) 
	{ 
		m_eulerAngles = eulerAngles;
		m_rotation = Converter::toQtQuat(glm::quat(glm::radians(Converter::toGLMVec3(m_eulerAngles)))); 
	}
	inline void setRotation(float x, float y, float z) 
	{ 
		m_eulerAngles = vec3(x, y, z);
		m_rotation = Converter::toQtQuat(glm::quat(glm::radians(Converter::toGLMVec3(m_eulerAngles)))); 
	}
	inline void setEulerAngleX(float x) 
	{ 
		m_eulerAngles.setX(x); 
		m_rotation = Converter::toQtQuat(glm::quat(glm::radians(Converter::toGLMVec3(m_eulerAngles)))); 
	}
	inline void setEulerAngleY(float y) 
	{ 
		m_eulerAngles.setY(y); 
		m_rotation = Converter::toQtQuat(glm::quat(glm::radians(Converter::toGLMVec3(m_eulerAngles)))); 
	}
	inline void setEulerAngleZ(float z) 
	{ 
		m_eulerAngles.setZ(z); 
		m_rotation = Converter::toQtQuat(glm::quat(glm::radians(Converter::toGLMVec3(m_eulerAngles)))); 
	}

	inline void rotate(const quat& delta) { m_rotation = delta * m_rotation; }

	/// Scale Access
	inline const vec3& getScale() const { return m_scale; }
	inline void setScale(const vec3& scale) { m_scale = scale; } 
	inline void setScale(float x, float y, float z) { m_scale = vec3(x, y, z); }
	inline void setScale(float factor) { m_scale = vec3(factor, factor, factor); }
	inline void setScaleX(float x) { m_scale.setX(x); }
	inline void setScaleY(float y) { m_scale.setY(y); }
	inline void setScaleZ(float z) { m_scale.setZ(z); }

	/// Model Matrix Access
	inline mat4 getTransformMatrix() const
	{
		mat4 m;
		m.translate(m_position);
		m.rotate(m_rotation);
		m.scale(m_scale);

		return m;
	}

	/// Reset
	inline void reset()
	{
		setPosition(Vector3::ZERO);
		setRotation(Vector3::ZERO);
		setScale(Vector3::UNIT_SCALE);
	}

	inline void toIdentity()
	{
		m_position = Vector3::ZERO;
		m_eulerAngles = Vector3::ZERO;
		m_scale = Vector3::UNIT_SCALE;
		m_rotation = Quaternion::IDENTITY;
	}

	void inverse();
	Transform inversed() const;

	vec3 operator*(const vec3& pos) const;
	Transform operator*(const Transform& transform) const;
	Transform& operator=(const Transform& other);

private:
	vec3 m_position;
	vec3 m_eulerAngles;
	quat m_rotation;
	vec3 m_scale;
};

