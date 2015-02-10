#include "Transform.h"
#include <Utility/Math.h>


Transform::Transform()
{}

Transform::Transform( const Transform& other )
{
	m_Translation = other.m_Translation;
	m_Rotation = other.m_Rotation;
}

Transform::Transform( const vec3& translation, const quat& rotation )
{
	m_Translation = translation;
	m_Rotation = rotation;
}

void Transform::Inverse()
{
	m_Rotation = m_Rotation.conjugate();
	m_Translation = m_Rotation.rotatedVector(-m_Translation);
}

Transform Transform::InverseOther() const
{
	Transform other(*this);
	other.Inverse();
	return other;
}

vec3 Transform::operator*( const vec3& vector ) const
{
	return m_Rotation.rotatedVector(vector) + m_Translation;
}

Transform Transform::operator*( const Transform& transform ) const
{
	return Transform(m_Rotation.rotatedVector(transform.GetTranslation()) + m_Translation, 
		m_Rotation * transform.GetRotation());
}

Transform& Transform::operator=( const Transform& other )
{
	if ( this == &other )
		return (*this);

	m_Translation = other.m_Translation;
	m_Rotation = other.m_Rotation;

	return (*this);
}
