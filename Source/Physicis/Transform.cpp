#include "Transform.h"
#include <Utility/Math.h>


Transform::Transform()
{}

Transform::Transform( const Transform& other )
{
	m_translation = other.m_translation;
	m_rotation = other.m_rotation;
}

Transform::Transform( const vec3& translation, const quat& rotation )
{
	m_translation = translation;
	m_rotation = rotation;
}

void Transform::inverse()
{
	m_rotation = m_rotation.conjugate();
	m_translation = m_rotation.rotatedVector(-m_translation);
}

Transform Transform::inversed() const
{
	Transform other(*this);
	other.inverse();
	return other;
}

vec3 Transform::operator*( const vec3& vector ) const
{
	return m_rotation.rotatedVector(vector) + m_translation;
}

Transform Transform::operator*( const Transform& transform ) const
{
	return Transform(m_rotation.rotatedVector(transform.getTranslation()) + m_translation, 
		m_rotation * transform.getRotation());
}

Transform& Transform::operator=( const Transform& other )
{
	if ( this == &other )
		return (*this);

	m_translation = other.m_translation;
	m_rotation = other.m_rotation;

	return (*this);
}
