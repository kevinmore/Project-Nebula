#pragma once
#include <Utility/EngineCommon.h>
#include <Utility/Math.h>
#include <QColor>

using namespace Math;
class Particle
{
public:
	Particle();
	~Particle();

	// dynamics properties
	float m_mass;
	vec3 m_force;
	vec3 m_position;
	vec3 m_velocity;
	vec3 m_acceleration;
	float m_rotate;  // Determines the amount of rotation to apply to the particle¡¯s local z-axis.

	// rendering properties
	QColor m_color;
	float m_size;    // Determines how large the particle will be and this is measured in world-coordinates, not screen-coordinates.

	// life properties
	float m_age; // The duration in seconds since the particle was emitted.
	float m_lifeTime; // How long the particle will live for. When the particle¡¯s age has exceeded it¡¯s lifetime, it is considered ¡°dead¡± and will not be rendered until it is re-emitted.
};

typedef QVector<Particle> ParticleBuffer;

