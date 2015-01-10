#pragma once
#include <QtCore>
#include <Scene/GameObject.h>
#include <Utility/EngineCommon.h>

/************************************************************************/
/*                           IO Streams                                 */
/************************************************************************/

// GameObject
// Order: Position, Rotation, Scaling
QDataStream& operator << (QDataStream& out, GameObject& go)
{
	out << go.position() << go.rotation() << go.scale();
	return out;
}

QDataStream& operator >> (QDataStream& in, GameObject& go)
{
	vec3 pos, rot, scale;
	in >> pos >> rot >> scale;

	go.setPosition(pos);
	go.setRotation(rot);
	go.setScale(scale);

	return in;
}

QDataStream& operator << (QDataStream& out, GameObject* go)
{
	out << go->position() << go->rotation() << go->scale();
	return out;
}

QDataStream& operator >> (QDataStream& in, GameObject* go)
{
	vec3 pos, rot, scale;
	in >> pos >> rot >> scale;

	go->setPosition(pos);
	go->setRotation(rot);
	go->setScale(scale);

	return in;
}