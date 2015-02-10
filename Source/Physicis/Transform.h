#pragma once
#include <Utility/EngineCommon.h>
class Transform

{
public:
	Transform();
	Transform(const Transform& other);
	Transform(const vec3& translation, const quat& rotation);

	const vec3& getTranslation() const { return m_translation; }
	const quat& getRotation() const { return m_rotation; }
	vec3& getTranslation() { return m_translation; }
	quat& getRotation() { return m_rotation; }
	void inverse();
	Transform inversed() const;

	vec3 operator*(const vec3& vector) const;
	Transform operator*(const Transform& transform) const;
	Transform& operator=(const Transform& other);

private:
	vec3 m_translation;
	quat m_rotation;
};

