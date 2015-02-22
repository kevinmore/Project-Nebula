#include "Particle.h"

Particle::Particle()
	: m_mass(0.001f),
	  m_force(Vector3D::ZERO),
	  m_position(Vector3D::ZERO),
	  m_velocity(Vector3D::ZERO),
	  m_acceleration(Vector3D::ZERO),
	  m_rotate(0.0f),
	  m_color(QColor()),
	  m_age(0.0f),
	  m_lifeTime(0.0f)
{
}


Particle::~Particle()
{
}
