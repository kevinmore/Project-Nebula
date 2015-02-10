#pragma once
#include <Utility/EngineCommon.h>
class Transform

{
public:
	Transform();
	Transform(const Transform& other);
	Transform(const vec3& translation, const quat& rotation);

	const vec3& GetTranslation() const { return m_Translation; }
	const quat& GetRotation() const { return m_Rotation; }
	vec3& GetTranslation() { return m_Translation; }
	quat& GetRotation() { return m_Rotation; }
	void Inverse();
	Transform InverseOther() const;

	vec3 operator*(const vec3& vector) const;
	Transform operator*(const Transform& transform) const;
	Transform& operator=(const Transform& other);

private:
	vec3 m_Translation;
	quat m_Rotation;
};

