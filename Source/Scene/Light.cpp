#include "Light.h"
#include <Utility/Math.h>
using namespace Math;
const float INNER_ANGLE = 30.0f; // 30deg = 0.6981rad = cos(2PI/9) = 0.76604;
const float OUTER_ANGLE = 40.0f; // 40deg = 0.5235rad = cos(1PI/6) = 0.86602;

Light::Light(const QString& name) :
	Component(),
	m_name(name),
	m_type(PointLight),
	m_position(Math::Vector3::ZERO),
	m_direction(Math::Vector3::NEGATIVE_UNIT_Z),
	m_color(Qt::white),
	m_constantAttenuation(1.0f),
	m_linearAttenuation(0.0f),
	m_quadraticAttenuation(0.0f),
	m_spotFalloff(1.0f),
	m_spotInnerAngle(cosf(static_cast<float>(M_PI)*(INNER_ANGLE/180.0f))),
	m_spotOuterAngle(cosf(static_cast<float>(M_PI)*(OUTER_ANGLE/180.0f))),
	m_intensity(1.0f)
{}

Light::~Light() {}

void Light::setType(LightType type)
{
	m_type = type;
}

Light::LightType Light::type() const
{
	return m_type;
}

void Light::setIntensity(float intensity)
{
	m_intensity = intensity;
}

float Light::intensity() const
{
	return m_intensity;
}

void Light::setColor(const QColor& color)
{
	if(color.alpha())
		qWarning("The alpha channel of light color should be equal to 0");

	m_color  = color;
}

void Light::setColor(float r, float g, float b)
{
	m_color.setRgbF(r, g, b, 0.0);
}

const QColor& Light::color() const
{
	return m_color;
}

void Light::setPosition(const QVector3D& position)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use position property");

	m_position = position;
}

void Light::setPosition(float x, float y, float z)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use position property");

	m_position.setX(x);
	m_position.setY(y);
	m_position.setZ(z);
}

const QVector3D& Light::position() const
{
	return m_position;
}

void Light::setDirection(const QVector3D& direction)
{
	if(m_type == PointLight)
		qWarning("Point lights not use direction property");

	m_direction = direction;
}

void Light::setDirection(float x, float y, float z)
{
	if(m_type == PointLight)
		qWarning("Point lights not use direction property");

	m_direction.setX(x);
	m_direction.setY(y);
	m_direction.setZ(z);
}

const QVector3D& Light::direction() const
{
	return m_direction.normalized();
}

void Light::setAttenuation(float constantFactor,
	float linearFactor,
	float quadraticFactor)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use attenuation property");

	m_constantAttenuation  = constantFactor;
	m_linearAttenuation    = linearFactor;
	m_quadraticAttenuation = quadraticFactor;
}

void Light::setConstantAttenuation(float constantFactor)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use constant attenuation property");

	m_constantAttenuation = constantFactor;
}

void Light::setLinearAttenuation(float linearFactor)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use linear attenuation property");

	m_linearAttenuation = linearFactor;
}

void Light::setQuadraticAttenuation(float quadraticFactor)
{
	if(m_type == DirectionalLight)
		qWarning("Directional lights not use quadratic attenuation property");

	m_quadraticAttenuation = quadraticFactor;
}

float Light::constantAttenuation() const
{
	return m_constantAttenuation;
}

float Light::linearAttenuation() const
{
	return m_linearAttenuation;
}

float Light::quadraticAttenuation() const
{
	return m_quadraticAttenuation;
}

void Light::setSpotFalloff(float falloff)
{
	if(m_type != SpotLight)
		qWarning("Only spotlights can set falloff property");

	m_spotFalloff = falloff;
}

void Light::setSpotInnerAngle(float innerAngle)
{
	if(m_type != SpotLight)
		qWarning("Only spotlights can set inner angle value");

	m_spotInnerAngle = cosf(static_cast<float>(M_PI)*(innerAngle/180.0f));
}

void Light::setSpotOuterAngle(float outerAngle)
{
	if(m_type != SpotLight)
		qWarning("Only spotlights can set outer angle value");

	m_spotOuterAngle = cosf(static_cast<float>(M_PI)*(outerAngle/180.0f));
}

float Light::spotFallOff() const
{
	return m_spotFalloff;
}

float Light::spotInnerAngle() const
{
	return m_spotInnerAngle;
}

float Light::spotOuterAngle() const
{
	return m_spotOuterAngle;
}
