/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>

// Serialization dependencies
#include <Common/Visualize/Serialize/hkDisplaySerializeOStream.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#include <Common/Visualize/hkDisplayGeometryBuilder.h>
#include <Common/Visualize/hkServerDebugDisplayHandler.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeIStream.h>
#include <Common/Visualize/Shape/hkDisplayGeometry.h>

hkServerDebugDisplayHandler::hkServerDebugDisplayHandler(hkDisplaySerializeOStream* outStream, hkDisplaySerializeIStream* inStream )
	: hkProcess( false )
	, m_continueData( HK_NULL )
	, m_simpleShapesPerPart( 0 )
	, m_simpleShapesPerFrame( 0 )
	, m_sendHashes( false )
{
	m_outStream = outStream;
	m_inStream = inStream;
	m_outstreamLock = new hkCriticalSection(1000); // usually no contention
}

hkServerDebugDisplayHandler::~hkServerDebugDisplayHandler()
{
	delete m_outstreamLock;
	// If the connection between the client and server has failed while there were shapes awaiting requests, we need to free
	// the additional references here.
	const int numPending = m_geometriesAwaitingRequests.getSize();
	for ( int i = 0; i < numPending; ++i )
	{
		m_geometriesAwaitingRequests[i].m_builder->removeReference();
		m_geometriesAwaitingRequests[i].m_source->removeReference();
	}

	const int numAwaitingDeparture = m_geometriesAwaitingDeparture.getSize();
	for ( int i = 0; i < numAwaitingDeparture; ++i )
	{
		m_geometriesAwaitingDeparture[i].m_builder->removeReference();
		m_geometriesAwaitingDeparture[i].m_source->removeReference();
	}	

	if ( m_continueData )
	{
		m_continueData->removeReference();
	}
}

static hkUint32 _getGeometryByteSize(const hkArrayBase<hkDisplayGeometry*>& geometries)
{
	hkUint32 bytes = 4; // numGeoms size
	for(int i = 0; i < geometries.getSize(); i++)
	{
		bytes += hkDisplaySerializeOStream::computeBytesRequired(geometries[i]);
	}
	return bytes;
}

void hkServerDebugDisplayHandler::sendGeometryData(const hkArrayBase<hkDisplayGeometry*>& geometries)
{
	m_outstreamLock->enter();
	{
		m_outStream->write32(geometries.getSize());
		for(int i = 0; i < geometries.getSize(); i++)
		{
			m_outStream->writeDisplayGeometry(geometries[i]);
		}
	}
	m_outstreamLock->leave();
}

hkResult hkServerDebugDisplayHandler::addGeometry(const hkArrayBase<hkDisplayGeometry*>& geometries, const hkTransform& transform, hkUlong id, int tag, hkUlong shapeIdHint, hkGeometry::GeometryType geomType)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + _getGeometryByteSize(geometries) + 7*4 + 8 + 4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32(packetSize);
			// send a serialized version of the displayObject
			if( geomType == hkGeometry::GEOMETRY_DYNAMIC_VERTICES )
			{
				m_outStream->write8u(hkVisualDebuggerProtocol::HK_ADD_DYNAMIC_VERTICES_GEOMETRY);
			}
			else
			{
				m_outStream->write8u(hkVisualDebuggerProtocol::HK_ADD_GEOMETRY);
			}
			sendGeometryData(geometries);
			m_outStream->writeTransform(transform);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return  streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::addGeometryInstance(hkUlong originalInstanceID, const hkTransform& transform, hkUlong id, int tag, hkUlong shapeIdHint)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + 8 + 7*4 + 8 + 4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_ADD_GEOMETRY_INSTANCE);
			m_outStream->write64u(originalInstanceID);
			m_outStream->writeTransform(transform);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::setGeometryPickable( hkBool isPickable, hkUlong id, int tag )
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if(m_outStream)
		{
			const int packetSize = 1 + 1 + 8 + 4;
			const hkUint64 longId = (hkUint64)id;
			m_outStream->write32(packetSize);

			m_outStream->write8u(hkVisualDebuggerProtocol::HK_SET_GEOMETRY_PICKABLE);
			m_outStream->write8u(isPickable);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::displayGeometry(const hkArrayBase<hkDisplayGeometry*>& geometries, const hkTransform& transform, hkColor::Argb color, int id, int tag)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if	(m_outStream)
		{
			const int packetSize = 1 + _getGeometryByteSize(geometries) + 7*4 + 4 + 8 + 4;
			const hkUint64 longId = (hkUint64)id;
			m_outStream->write32(packetSize);

			// send a serialized version of the displayObject
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_DISPLAY_GEOMETRY_WITH_TRANSFORM);
			sendGeometryData(geometries);
			m_outStream->writeTransform(transform);
			m_outStream->write32u(color);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::displayGeometry(const hkArrayBase<hkDisplayGeometry*>& geometries, hkColor::Argb color, int id, int tag)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if	(m_outStream)
		{
			const int packetSize = 1 + _getGeometryByteSize(geometries) + 4 + 8 + 4;
			const hkUint64 longId = (hkUint64)id;
			m_outStream->write32(packetSize);

			// send a serialized version of the displayObject
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_DISPLAY_GEOMETRY);
			sendGeometryData(geometries);
			m_outStream->write32u(color);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}


hkResult hkServerDebugDisplayHandler::setGeometryColor(hkColor::Argb color, hkUlong id, int tag)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + 4 + 8 + 4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_SET_COLOR_GEOMETRY);
			m_outStream->write32u(color);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::setGeometryTransparency(float alpha, hkUlong id, int tag)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + 4 + 8 + 4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_SET_TRANSPARENCY_GEOMETRY);
			m_outStream->writeFloat32(alpha);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::updateGeometry(const hkTransform& transform, hkUlong id, int tag)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		// update a display object identified by id
		if (m_outStream)
		{
			const int packetSize = 1 + (7*4) + 8 + 4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_UPDATE_GEOMETRY);
			m_outStream->writeTransform(transform); // 7 * float
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::updateGeometryVerts(hkUlong geomID, const hkArray<hkVector4>& newVerts, int tag)
{
	bool streamOK;
	m_outstreamLock->enter();


	const int packetSize = 1 + 8 + 4 + newVerts.getSize() * 12 + 4;
	const hkUint64 longId = (hkUint64)geomID;

	m_outStream->write32u(packetSize);
	m_outStream->write8u(hkVisualDebuggerProtocol::HK_UPDATE_GEOMETRY_VERTS);
	m_outStream->write64u(longId);
	m_outStream->write32u(newVerts.getSize());
	for( int i = 0; i < newVerts.getSize(); i++ )
	{
		m_outStream->writeQuadVector4( newVerts[i] );
	}
	m_outStream->write32(tag);
	streamOK = (m_outStream && m_outStream->isOk());
	m_outstreamLock->leave();
	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::updateGeometry( const hkMatrix4& transform, hkUlong id, int tag )
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + 8 + 16 * sizeof(hkFloat32) + 4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32u(packetSize);			
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_UPDATE_GEOMETRY_WITH_SCALE);
			m_outStream->write64u(longId);						
			m_outStream->writeArrayFloat32(&transform.getColumn(0)(0), 16);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;	
}

hkResult hkServerDebugDisplayHandler::skinGeometry(hkUlong* ids, int numIds, const hkMatrix4* poseModel, int numPoseModel, const hkMatrix4& worldFromModel, int tag )
{
	bool streamOK;

	HK_TIMER_BEGIN("send skin", HK_NULL);

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + 4 + (8 * numIds) + 4 + ((numPoseModel + 1) * 16 * sizeof(hkFloat32)) + 4;			

			m_outStream->write32u(packetSize);			
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_SKIN_GEOMETRY);
			m_outStream->write32u(numIds);
			for( int i = 0; i < numIds; ++i )
			{
				const hkUint64 longId = (hkUint64)ids[i];
				m_outStream->write64u(longId);
			}			
			m_outStream->write32u(numPoseModel);
			m_outStream->writeArrayFloat32(&poseModel[0].getColumn(0)(0), 16 * numPoseModel);			
			m_outStream->writeArrayFloat32(&worldFromModel.getColumn(0)(0), 16);
			m_outStream->write32(tag);

			HK_MONITOR_ADD_VALUE("bytes", float(packetSize), HK_MONITOR_TYPE_INT);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	HK_TIMER_END();

	return streamOK ? HK_SUCCESS : HK_FAILURE;	
}

hkResult hkServerDebugDisplayHandler::removeGeometry(hkUlong id, int tag, hkUlong shapeIdHint)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + 8 + 4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32u(packetSize);
			// remove a display object identified by id
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_REMOVE_GEOMETRY);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::updateCamera(const hkVector4& from, const hkVector4& to, const hkVector4& up, hkReal nearPlane, hkReal farPlane, hkReal fov, const char* name)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int length = hkString::strLen(name);
			const int packetSize = 1 + 3*3*4 + 3*4 + 2 + length;

			m_outStream->write32u(packetSize);
			// update a display object identified by id
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_UPDATE_CAMERA);
			m_outStream->writeQuadVector4(from); //3*float
			m_outStream->writeQuadVector4(to);
			m_outStream->writeQuadVector4(up);
			m_outStream->writeFloat32(hkFloat32(nearPlane));
			m_outStream->writeFloat32(hkFloat32(farPlane));
			m_outStream->writeFloat32(hkFloat32(fov));
			// send the name
			m_outStream->write16u( hkUint16(length) );
			m_outStream->writeRaw(name, length);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::displayPoint(const hkVector4& position, hkColor::Argb color, int id, int tag)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + 3*4 + 4 + 8 + 4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_DISPLAY_POINT);
			m_outStream->writeQuadVector4(position);
			m_outStream->write32u(color);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::displayLine(const hkVector4& start, const hkVector4& end, hkColor::Argb color, int id, int tag)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + 2*3*4 + 4 + 8 + 4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_DISPLAY_LINE);
			m_outStream->writeQuadVector4(start);
			m_outStream->writeQuadVector4(end);
			m_outStream->write32u(color);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::displayTriangle(const hkVector4& a, const hkVector4& b, const hkVector4& c, hkColor::Argb color, int id, int tag)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			hkVector4 p[4];
			p[0] = a;
			p[1] = b;
			p[2] = c;
			p[3] = a;
			for( int i = 0; i < 3; ++i )
			{
				const int packetSize = 1 + 2*3*4 + 4 + 8 + 4;
				const hkUint64 longId = (hkUint64)id;

				m_outStream->write32u(packetSize);
				m_outStream->write8u(hkVisualDebuggerProtocol::HK_DISPLAY_LINE);
				m_outStream->writeQuadVector4(p[i]);
				m_outStream->writeQuadVector4(p[i+1]);
				m_outStream->write32u(color);
				m_outStream->write64u(longId);
				m_outStream->write32(tag);
			}
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::displayText(const char* text, hkColor::Argb color, int id, int tag)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int length = hkMath::min2(hkString::strLen(text), 65535);
			const int packetSize = 1 + 2 + length + 4 + 8 + 4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_DISPLAY_TEXT);
			m_outStream->write16u((unsigned short)length);
			m_outStream->writeRaw(text, length);
			m_outStream->write32(color);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::display3dText(const char* text, const hkVector4& pos, hkColor::Argb color, int id, int tag)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int length = hkMath::min2(hkString::strLen(text), 65535);
			const int packetSize = 1 + 2 + length + 4 + 8 + 4 + 3*4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_DISPLAY_TEXT_3D);
			m_outStream->write16u((unsigned short)length);
			m_outStream->writeRaw(text, length);
			m_outStream->write32(color);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
			m_outStream->writeFloat32(hkFloat32(pos(0)));
			m_outStream->writeFloat32(hkFloat32(pos(1)));
			m_outStream->writeFloat32(hkFloat32(pos(2)));
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}


hkResult hkServerDebugDisplayHandler::displayPoint2d(const hkVector4& position, hkColor::Argb color, int id, int tag)
{
	return HK_SUCCESS;
}

hkResult hkServerDebugDisplayHandler::displayLine2d(const hkVector4& start, const hkVector4& end, hkColor::Argb color, int id, int tag)
{
	return HK_SUCCESS;
}

hkResult hkServerDebugDisplayHandler::displayTriangle2d(const hkVector4& a, const hkVector4& b, const hkVector4& c, hkColor::Argb color, int id, int tag)
{
	return HK_SUCCESS;
}

hkResult hkServerDebugDisplayHandler::displayText2d(const char* text, const hkVector4& pos, hkReal sizeScale, hkColor::Argb color, int id, int tag)
{
	return HK_SUCCESS;
}

hkResult hkServerDebugDisplayHandler::displayAnnotation(const char* text, int id, int tag)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int length = hkMath::min2(hkString::strLen(text), 65535);
			const int packetSize = 1 + 2 + length + 8 + 4;
			const hkUint64 longId = (hkUint64)id;

			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_DISPLAY_ANNOTATION);
			m_outStream->write16u((unsigned short)length);
			m_outStream->writeRaw(text, length);
			m_outStream->write64u(longId);
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::displayBone(const hkVector4& a, const hkVector4& b, const hkQuaternion& orientation, hkColor::Argb color, int tag )
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{			
			const int packetSize = 1 + 12 + 12 + 16 + 4 + 4;

			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_DISPLAY_BONE);
			m_outStream->writeQuadVector4(a);
			m_outStream->writeQuadVector4(b);
			m_outStream->writeArrayFloat32(&orientation.m_vec(0), 4);			
			m_outStream->write32(color);						
			m_outStream->write32(tag);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

hkResult hkServerDebugDisplayHandler::sendMemStatsDump(const char* data, int length)
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + 4 + length;

			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_SEND_MEMSTATS_DUMP);
			m_outStream->write32(length);
			m_outStream->writeRaw(data, length);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}


hkResult hkServerDebugDisplayHandler::holdImmediate()
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1;

			m_outStream->write32u(packetSize);
			m_outStream->write8u(hkVisualDebuggerProtocol::HK_HOLD_IMMEDIATE);
		}
		streamOK = (m_outStream && m_outStream->isOk());
	}
	m_outstreamLock->leave();

	return streamOK ? HK_SUCCESS : HK_FAILURE;
}

static hkUint8 _serverDebugDisplayHandler_cmds[] = { 
	hkVisualDebuggerProtocol::COMMAND_REQUEST_GEOMETRY_WITH_HASH,
	hkVisualDebuggerProtocol::COMMAND_DO_NOT_REQUEST_GEOMETRY_WITH_HASH,
	hkVisualDebuggerProtocol::COMMAND_CLIENT_DISPLAY_HANDLER_SETTINGS
};

void hkServerDebugDisplayHandler::getConsumableCommands( hkUint8*& commands, int& numCommands )
{
	commands = _serverDebugDisplayHandler_cmds;
	numCommands	= sizeof( _serverDebugDisplayHandler_cmds );
}

void hkServerDebugDisplayHandler::consumeCommand( hkUint8 command  )
{
	switch( command )
	{
	case hkVisualDebuggerProtocol::COMMAND_CLIENT_DISPLAY_HANDLER_SETTINGS:
		{
			hkStringPtr compilerString;
			m_simpleShapesPerPart = m_inStream->read32u();
			m_sendHashes = m_inStream->read8u() ? true : false;
			m_simpleShapesPerFrame = m_inStream->read32u();
			break;
		}
	case hkVisualDebuggerProtocol::COMMAND_REQUEST_GEOMETRY_WITH_HASH:
		{
			hkDebugDisplayHandler::Hash hash;
			m_inStream->readHash( hash );
			processGeometryWithHashRequested( hash );
			break;
		}
	case hkVisualDebuggerProtocol::COMMAND_DO_NOT_REQUEST_GEOMETRY_WITH_HASH:
		{
			hkDebugDisplayHandler::Hash hash;
			m_inStream->readHash( hash );
			processGeometryWithHashNotRequested( hash );
			break;
		}
	default:
		break;
	}
}

hkResult hkServerDebugDisplayHandler::useGeometryForHash( const hkArray<hkDisplayGeometry*>& geometries, const Hash& hash, hkBool final )
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + sizeof(Hash) + _getGeometryByteSize(geometries);

			m_outStream->write32( packetSize );
			// send a serialized version of the displayObject
			m_outStream->write8u( (hkUint8) ( final ? hkVisualDebuggerProtocol::HK_GEOMETRY_FOR_HASH_FINAL : hkVisualDebuggerProtocol::HK_GEOMETRY_FOR_HASH_PART ) );
			m_outStream->writeHash( hash );
			sendGeometryData( geometries );
		}
		streamOK = ( m_outStream && m_outStream->isOk() );
	}
	m_outstreamLock->leave();

	return  streamOK ? HK_SUCCESS : HK_FAILURE;
}


hkResult hkServerDebugDisplayHandler::addGeometryHash( const hkReferencedObject* source, hkDisplayGeometryBuilder* builder, const Hash& hash, const hkAabb& aabb, hkColor::Argb color, const hkTransform& transform, hkUlong id, int tag )
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if (m_outStream)
		{
			const int packetSize = 1 + sizeof(Hash) + 6*4 + 4 + 7*4 + 8 + 4;
			const hkUint64 longId = (hkUint64) id;

			m_outStream->write32( packetSize );
			// send a serialized version of the displayObject
			m_outStream->write8u( hkVisualDebuggerProtocol::HK_ADD_GEOMETRY_HASH );
			m_outStream->writeHash( hash );
			m_outStream->writeAabb( aabb );
			m_outStream->write32u( color );
			m_outStream->writeTransform( transform );
			m_outStream->write64u( longId );
			m_outStream->write32( tag );
		}
		streamOK = ( m_outStream && m_outStream->isOk() );
	}
	m_outstreamLock->leave();

	// Store information about the hash
	{
		// Do we have a source and builder already for this hash?
		HashCountMap::Iterator iterator = m_hashCountMap.findOrInsertKey( hash, 0 );
		const int count = m_hashCountMap.getValue( iterator );
		if ( count == 0 )
		{
			// No.
			// Add the info to the geometries awaiting requests array.
			UnbuiltGeometryInfo& newEntry = m_geometriesAwaitingRequests.expandOne();
			newEntry.m_hash = hash;
			newEntry.m_source = source;
			newEntry.m_builder = builder;
			// Keep the source and builder alive.
			source->addReference();
			builder->addReference();
		}
		m_hashCountMap.setValue( iterator, count + 1 );
	}
	return  streamOK ? HK_SUCCESS : HK_FAILURE;
}

void hkServerDebugDisplayHandler::processGeometryWithHashRequested( const Hash& hash )
{
	const int index = findIndexForGeometryAwaitingRequest( hash );

	HK_ASSERT2( 0xa8b7d6de, index != -1, "Can't find source object for geometry requested command" );

	// Move the info from one array to the other.
	m_geometriesAwaitingDeparture.expandOne() = m_geometriesAwaitingRequests[index];
	m_geometriesAwaitingRequests.removeAt( index );
}

void hkServerDebugDisplayHandler::processGeometryWithHashNotRequested( const hkDebugDisplayHandler::Hash& hash )
{
	HashCountMap::Iterator iterator = m_hashCountMap.findKey( hash );
	HK_ASSERT2( 0xea8b291e, m_hashCountMap.isValid( iterator ), "Count for source object invalid." );
	const int newCount = m_hashCountMap.getValue( iterator ) - 1;
	HK_ASSERT2( 0xea8b291f, newCount >= 0, "Count for source object invalid." );
	m_hashCountMap.setValue( iterator, newCount );
	if ( newCount == 0 )
	{
		// The info must be in the awaiting requests list.
		const int index = findIndexForGeometryAwaitingRequest( hash );
		HK_ASSERT2( 0xa8b7d6de, index != -1, "Can't find source object for geometry not requested command" );
		UnbuiltGeometryInfo& entry = m_geometriesAwaitingRequests[index];
		entry.m_builder->removeReference();
		entry.m_source->removeReference();
		m_geometriesAwaitingRequests.removeAt( index );
		// Remove the map entry.
		m_hashCountMap.remove( iterator );
	}
}

int hkServerDebugDisplayHandler::findIndexForGeometryAwaitingRequest( const hkDebugDisplayHandler::Hash& hash ) const
{
	const int numPending = m_geometriesAwaitingRequests.getSize();
	for ( int i = 0; i < numPending; ++i )
	{
		if ( hash == m_geometriesAwaitingRequests[i].m_hash )
		{
			return i;
		}
	}
	return -1;
}

void hkServerDebugDisplayHandler::buildAndSendGeometries()
{
	// This just guarantees the loop condition in the case when we're not limiting by frame.
	int numSimpleShapes = m_simpleShapesPerFrame ? m_simpleShapesPerFrame : 1;

	const int numAwaitingDeparture = m_geometriesAwaitingDeparture.getSize();

	int index = 0;
	hkBool completedShape = true;

	while ( ( index < numAwaitingDeparture ) && numSimpleShapes )
	{
		UnbuiltGeometryInfo& info = m_geometriesAwaitingDeparture[index];

		hkInplaceArray<hkDisplayGeometry*,8> displayGeometries;

		// Do we need to split up the shape?
		// We take account of the fact that the settings may have changed while we still have a
		// partially built shape to process by checking the m_continueData.
		if ( m_simpleShapesPerFrame || m_simpleShapesPerPart || m_continueData )
		{
			if ( m_continueData == HK_NULL )
			{
				m_continueData = info.m_builder->getInitialContinueData( info.m_source );
			}

			// This should be big enough for the rest of the shape if the only reason
			// we're using buildPartialDisplayGeometries is that we have a continue data from the
			// previous frame.
			int numSimpleShapesAvailable = 100000000;
			{
				// Limit by frame, if necessary.
				if ( m_simpleShapesPerFrame )
				{
					numSimpleShapesAvailable = hkMath::min2( numSimpleShapesAvailable, numSimpleShapes );
				}
				// Limit by part, if necessary.
				if ( m_simpleShapesPerPart )
				{
					numSimpleShapesAvailable = hkMath::min2( numSimpleShapesAvailable, m_simpleShapesPerPart );
				}
			}

			// This will get changed by the builder.
			int numSimpleShapesThisShape = numSimpleShapesAvailable;

			// Did we have enough room to finish building and sending this shape?
			completedShape = info.m_builder->buildPartialDisplayGeometries( info.m_source, numSimpleShapesThisShape, m_continueData, displayGeometries );

			// If we are limiting the number of shapes per frame, we subtract the number we built here.
			if ( m_simpleShapesPerFrame )
			{
				numSimpleShapes -= ( numSimpleShapesAvailable - numSimpleShapesThisShape );
			}
		}
		else
		{
			// Build the whole shape.
			info.m_builder->buildDisplayGeometries( info.m_source, displayGeometries );
			completedShape = true;
		}

		for(int i = (displayGeometries.getSize() - 1); i >= 0; i--)
		{
			if( (displayGeometries[i]->getType() == HK_DISPLAY_CONVEX) &&
				(displayGeometries[i]->getGeometry() == HK_NULL) )
			{
				HK_REPORT("Unable to build some display geometry data");
				displayGeometries.removeAt(i);
			}
		}

		if ( completedShape )
		{
			m_continueData = HK_NULL;
			++index;
			useGeometryForHash( displayGeometries, info.m_hash, true );
			{
				HashCountMap::Iterator iterator = m_hashCountMap.findKey( info.m_hash );
				HK_ASSERT2( 0xea8b291e, m_hashCountMap.isValid( iterator ), "Count for source object invalid." );
				const int newCount = m_hashCountMap.getValue( iterator ) - 1;
				HK_ASSERT2( 0xea8b291f, newCount >= 0, "Count for source object invalid." );
				m_hashCountMap.setValue( iterator, newCount );
				if ( newCount == 0 )
				{
					info.m_builder->removeReference();
					info.m_source->removeReference();
					// Remove the map entry.
					m_hashCountMap.remove( iterator );
				}
				else
				{
					// Move it back into the awaiting requests list.
					m_geometriesAwaitingRequests.expandOne() = info;
				}
			}
		}
		else
		{
			useGeometryForHash( displayGeometries, info.m_hash, false );
		}

		for( int i = 0; i < displayGeometries.getSize(); ++i )
		{
			delete displayGeometries[i];
		}
	}

	// Remove those we've successfully built and sent.
	if ( index > 0 )
	{
		m_geometriesAwaitingDeparture.removeAtAndCopy( 0, index );
	}
}

void hkServerDebugDisplayHandler::step( hkReal frameTimeInMs )
{
	buildAndSendGeometries();
}

hkBool hkServerDebugDisplayHandler::doesSupportHashes() const
{
	return m_sendHashes;
}

hkResult hkServerDebugDisplayHandler::addGeometryPart( const hkArrayBase<hkDisplayGeometry*>& geometries, const hkTransform& transform, hkUlong id, int tag, hkUlong shapeIdHint, hkBool final )
{
	bool streamOK;

	m_outstreamLock->enter();
	{
		if ( m_outStream )
		{
			if ( final )
			{				
				// We only send the transform, id and tag with the final message.
				const int packetSize = 1 + _getGeometryByteSize( geometries ) + 7*4 + 8 + 4;
				const hkUint64 longId = (hkUint64) id;

				m_outStream->write32( packetSize );
				m_outStream->write8u( hkVisualDebuggerProtocol::HK_ADD_GEOMETRY_FINAL );
				sendGeometryData( geometries );
				m_outStream->writeTransform( transform );
				m_outStream->write64u( longId );
				m_outStream->write32( tag );
			}
			else
			{
				const int packetSize = 1 + _getGeometryByteSize(geometries);

				m_outStream->write32( packetSize );
				m_outStream->write8u( hkVisualDebuggerProtocol::HK_ADD_GEOMETRY_PART );
				sendGeometryData( geometries );
			}
		}
		streamOK = ( m_outStream && m_outStream->isOk() );
	}
	m_outstreamLock->leave();

	return  streamOK ? HK_SUCCESS : HK_FAILURE;	
}


hkResult hkServerDebugDisplayHandler::addGeometryLazily( const hkReferencedObject* source, hkDisplayGeometryBuilder* builder, const hkTransform& transform, hkUlong id, int tag, hkUlong shapeIdHint)
{
	hkResult result;
	if ( m_simpleShapesPerPart )
	{
		result = addGeometryInParts( source, builder, transform, id, tag, shapeIdHint );
	}
	else
	{
		result = hkDebugDisplayHandler::addGeometryLazily( source, builder, transform, id, tag, shapeIdHint );
	}
	return result;
}

hkResult hkServerDebugDisplayHandler::addGeometryInParts( const hkReferencedObject* source, hkDisplayGeometryBuilder* builder, const hkTransform& transform, hkUlong id, int tag, hkUlong shapeIdHint )
{
	HK_ASSERT2( 0x2a9d8fd2, m_simpleShapesPerPart > 0, "Client does not want shapes sent in parts" );
	hkReferencedObject* continueData = builder->getInitialContinueData( source );

	hkBool completedShape;
	bool streamOK = true;
	hkArray<hkDisplayGeometry*> displayGeometries;

	do
	{
		// Build up to m_simpleShapesPerPart this iteration.
		int numSimpleShapes = m_simpleShapesPerPart;
		completedShape = builder->buildPartialDisplayGeometries( source, numSimpleShapes, continueData, displayGeometries );

		for(int i = (displayGeometries.getSize() - 1); i >= 0; i--)
		{
			if( ( displayGeometries[i]->getType() == HK_DISPLAY_CONVEX ) &&
				( displayGeometries[i]->getGeometry() == HK_NULL ) )
			{
				HK_REPORT( "Unable to build display geometry from source" );
				displayGeometries.removeAt( i );
			}
		}

		if ( displayGeometries.getSize() )
		{
			streamOK = ( addGeometryPart( displayGeometries, transform, id, tag, shapeIdHint, completedShape ) == HK_SUCCESS );
		}

		if ( !streamOK )
		{
			break;
		}

		hkReferencedObject::removeReferences( displayGeometries.begin(), displayGeometries.getSize(), sizeof(hkDisplayGeometry*) );
		displayGeometries.clear();
	}
	while ( !completedShape );

	return streamOK ? HK_SUCCESS : HK_FAILURE;	
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
