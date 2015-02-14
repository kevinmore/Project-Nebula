#include "Transform.h"
#include <Utility/Math.h>

Transform::Transform()
	: m_position(Vector3::ZERO),
	  m_rotation(Quaternion::IDENTITY),
	  m_scale(Vector3::UNIT_SCALE)
{}

Transform::Transform( const Transform& other )
{
	m_position = other.m_position;
	m_rotation = other.m_rotation;
	m_scale    = other.m_scale;
	m_eulerAngles = other.m_eulerAngles;
}

Transform::Transform( const vec3& translation, const quat& rotation, const vec3& scale )
{
	m_position = translation;
	m_rotation = rotation;
	m_scale    = scale;

	m_eulerAngles = Quaternion::computeEularAngles(rotation);
}

void Transform::inverse()
{
	m_rotation = m_rotation.conjugate();
	m_position = m_rotation.rotatedVector(-m_position);
}

Transform Transform::inversed() const
{
	Transform other(*this);
	other.inverse();
	return other;
}

vec3 Transform::operator*( const vec3& pos ) const
{
	return m_rotation.rotatedVector(pos) + m_position;
}

Transform Transform::operator*( const Transform& transform ) const
{
	return Transform(m_rotation.rotatedVector(transform.getPosition()) + m_position, 
		m_rotation * transform.getRotation());
}

Transform& Transform::operator=( const Transform& other )
{
	if ( this == &other )
		return (*this);

	m_position = other.m_position;
	m_rotation = other.m_rotation;
	m_scale    = other.m_scale;
	m_eulerAngles = other.m_eulerAngles;

	return (*this);
}
