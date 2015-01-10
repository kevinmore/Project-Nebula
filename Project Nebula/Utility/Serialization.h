#pragma once
#include <QtCore>
#include <Scene/GameObject.h>
#include <Scene/Camera.h>
#include <Scene/Managers/ModelManager.h>

/************************************************************************/
/*                           IO Streams                                 */
/************************************************************************/

/*
* Game Object
*/
// Order: Position -> Rotation -> Scaling
QDataStream& operator << (QDataStream& out, GameObject& object)
{
	out << object.position() << object.rotation() << object.scale();
	return out;
}

QDataStream& operator >> (QDataStream& in, GameObject& object)
{
	vec3 pos, rot, scale;
	in >> pos >> rot >> scale;

	object.setPosition(pos);
	object.setRotation(rot);
	object.setScale(scale);

	return in;
}

QDataStream& operator << (QDataStream& out, GameObject* object)
{
	out << object->position() << object->rotation() << object->scale();
	return out;
}

QDataStream& operator >> (QDataStream& in, GameObject* object)
{
	vec3 pos, rot, scale;
	in >> pos >> rot >> scale;

	object->setPosition(pos);
	object->setRotation(rot);
	object->setScale(scale);

	return in;
}

/*
* Model Manager
*/
QDataStream& operator << (QDataStream& out, ModelManager& object)
{
	out << object.m_modelsInfo;
	return out;
}

QDataStream& operator >> (QDataStream& in, ModelManager& object)
{
	QVector<QPair<QString, GameObject*>> modelsInfo;
	in >> modelsInfo;
	object.m_modelsInfo = modelsInfo;

	return in;
}

QDataStream& operator << (QDataStream& out, QSharedPointer<ModelManager> object)
{
	out << object->m_modelsInfo;
	return out;
}

QDataStream& operator >> (QDataStream& in, QSharedPointer<ModelManager> object)
{
	QVector<QPair<QString, GameObject*>> modelsInfo;
	in >> modelsInfo;
	object->m_modelsInfo = modelsInfo;

	return in;
}