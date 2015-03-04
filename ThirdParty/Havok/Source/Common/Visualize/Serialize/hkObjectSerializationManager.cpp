/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/Serialize/hkObjectSerializationManager.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/Process/hkInspectProcess.h>

hkObjectSerializationManager::hkObjectSerializationManager( hkStreamWriter* streamWriter )
:	m_streamWriter(streamWriter),
	m_classClassId(0)
{
}

hkObjectSerializationManager::~hkObjectSerializationManager()
{
	// delete all of the fixups
	typedef hkPointerMap< hkUint64, hkArray<hkObjectSerialize::GlobalFixup>*>::Iterator It;

	for( It i = m_idToFixups.getIterator(); m_idToFixups.isValid(i); i = m_idToFixups.getNext(i) )
	{
		delete m_idToFixups.getValue(i);
	}
}

void hkObjectSerializationManager::setTargetRules(const hkStructureLayout::LayoutRules& rules )
{
	m_targetLayout = hkStructureLayout(rules);
}

void hkObjectSerializationManager::addDataType( hkUint64 id, hkUint64 classID, hkBool overwrite )
{
	TypeMapIterator iter = m_ptrToTypeMap.findKey( id );
	if ( m_ptrToTypeMap.isValid(iter) )
	{
		if (overwrite)
			m_ptrToTypeMap.setValue(iter, classID);
	}
	else
	{
		m_ptrToTypeMap.insert( id, classID );
	}
}

void hkObjectSerializationManager::readObject( hkStreamReader* s, void*& data, hkUint64& dataID, hkUint64& klassID )
{
	data = HK_NULL;
	hkUint32 dataSize = 0;
	hkArray<hkObjectSerialize::GlobalFixup>* globals = new hkArray<hkObjectSerialize::GlobalFixup>;
	klassID = 0;

	dataID = hkObjectSerialize::readObject( s, data, dataSize, *globals, klassID );

	// If we already have this object then return the existing one.
	
	
	

	
	
	{
		void* oldData = m_dataReg.getObjectData( dataID );

 		if ((oldData != HK_NULL) && ((klassID == m_classClassId)))
		{
			// deallocate the data and globals allocated above
			hkDeallocate<char>( reinterpret_cast<char*>( data ) );
			delete globals;

			// return the old data
			data = oldData;
			return;
		}
	}

	if ((dataID > 0) && data)
	{
		// remove from our pending list
		PendingReqIterator iter = m_pendingRequests.findKey( dataID );
		if ( m_pendingRequests.isValid(iter) )
		{
			m_pendingRequests.remove(iter);
		}
		
		// see if we have the type. It may be a virtual one that we have not requested
		// explicitly, so we will ask for it.
		addDataType(dataID, klassID, true); // overwrite/add proper specific klass id.
		{
			void* klassData = getObject(klassID);
			if (!klassData)
			{
				addDataType(klassID, m_classClassId, false); // we know it is a class.

				// ask for the class (unless this is the ClassClass right here, in which case the data and klass will have the same ID)
				if ( m_classClassId != dataID )
				{
					requestObject(klassID, true, false, false); // ask for it.
				}
			}
		}

		// add our data ptr
		m_dataReg.addObject(dataID, data, dataSize, klassID); // takes ownership of the data chunk

		// do some local fixups 
		char* dataPtr = reinterpret_cast<char*>( data );
		for (int g=0; g < globals->getSize(); ++g)
		{
			hkObjectSerialize::GlobalFixup& gf = (*globals)[g];
			addDataType(gf.toID, gf.toClassID, false); // so if the user wants to drill down, we already know the type (at least its base type)
			addDataType(gf.toClassID, m_classClassId, false );

			// this isn't 64 bit safe, but for now, add the obj id in instead of the ptr (which is prob null)
			*(void**)(dataPtr + gf.fromOffset)= (void*)(hkUlong)gf.toID;
		}

		// store fixup index
		if ( globals->getSize() == 0 )
		{
			delete globals;
		}
		else
		{
			if ( m_idToFixups.hasKey( dataID ) )
			{
				delete m_idToFixups.getWithDefault( dataID, HK_NULL );
			}

			m_idToFixups.insert( dataID, globals );
		}

		// if this object was live before, we have just updated it and wrote over ther live
		// ptrs .. so we need to be consistant and make it live still
		if (isLive(dataID))
		{
			makeLive(dataID, true); //force this id live update.
		}		
	}
	else
	{
		delete globals;
	}
}

void hkObjectSerializationManager::requestObject( hkUint64 id, bool recur, bool autoUpdate, bool forceRequest)
{
	if ( m_streamWriter != HK_NULL )
	{
		hkUint64 classID = getObjectType(id);
		if (classID)
		{
			//see if the request is actually pending (don't ask for it twice)
			PendingReqIterator iter = m_pendingRequests.findKey( id );
			if ( m_pendingRequests.isValid(iter) )
			{
				int count = m_pendingRequests.getValue(iter);
				if ((!forceRequest) || (count < 10)) // ask more than 10 times and it will get the thing again.
				{
					m_pendingRequests.setValue(iter, count+1);
					return;
				}
				m_pendingRequests.setValue(iter, 1);// start again.
			}
			else
			{
				m_pendingRequests.insert( id, 1 );
			}
		
			hkDisplaySerializeOStream sendStream( m_streamWriter );

			// send creation command header
			sendStream.write8u(hkVisualDebuggerProtocol::HK_REQUEST_OBJECT);
			sendStream.write64(id);
			sendStream.write64(classID);
			hkUint8 flags = 0;
			if (recur) flags |= HK_INSPECT_FLAG_RECURSE;
			if (autoUpdate) flags |= HK_INSPECT_FLAG_AUTOUPDATE;
			sendStream.write8( flags ); // auto update flag etc			
		}
	}
}

void hkObjectSerializationManager::discardObject( hkUint64 id )
{
	m_dataReg.deleteObject( id );
	m_ptrToTypeMap.remove( id );

	hkArray<hkObjectSerialize::GlobalFixup>* fixups = m_idToFixups.getWithDefault( id, HK_NULL );
	delete fixups;
	m_idToFixups.remove( id );
}

bool hkObjectSerializationManager::makeLive( hkUint64 ptrID, bool forceUpdate )
{
	if (!forceUpdate && isLive(ptrID)) 
	{
		return true;
	}

	// check we have all the data required, and make them live too.
	char* thisPtr = reinterpret_cast<char*>( getObject(ptrID) );
	if (!thisPtr)
		return false;

	// to avoid any infinite recursion.
	int idIndex = m_liveIds.getSize();
	m_liveIds.pushBack(ptrID);

	hkArray<hkObjectSerialize::GlobalFixup>* globals = m_idToFixups.getWithDefault( ptrID, HK_NULL );
	for (int i=0; globals && (i < globals->getSize()); ++i)
	{
		hkObjectSerialize::GlobalFixup& gf = (*globals)[i];
		// fixup our use of that ptr.
		void* dataPtr = getObject(gf.toID);
		if ( !dataPtr || !makeLive(gf.toID, false) ) // never recurse the force update.
		{
			m_liveIds.removeAt(idIndex);
			return false;
		}
     
		*(void**)(thisPtr + gf.fromOffset) = dataPtr;
	}

	// a class will have offsets that are from the server (not the local client)
	// so we must recompute them if we want to use the class locally
	if ( getClassClassId() == getObjectType(ptrID) ) // is a hkClass
	{
		hkStructureLayout layout;
		hkPointerMap<const hkClass*,int> done;
		hkClass* kp = (hkClass*)thisPtr;
		layout.computeMemberOffsetsInplace( *kp, done, false );
	}

	return true;
}

static void _getEndianSwappedData(char* ptr, hkClassMember::Type type, char* buffer)
{
	switch(type)
	{
		case hkClassMember::TYPE_BOOL:
		case hkClassMember::TYPE_CHAR:
		case hkClassMember::TYPE_INT8:
		case hkClassMember::TYPE_UINT8:
			buffer[0] = ptr[0]; // straight through
			break;
		case hkClassMember::TYPE_INT16:
		case hkClassMember::TYPE_UINT16:
		case hkClassMember::TYPE_HALF:
			buffer[1] = ptr[0];
			buffer[0] = ptr[1];
			break;
		case hkClassMember::TYPE_INT32:
		case hkClassMember::TYPE_UINT32:
#ifndef HK_REAL_IS_DOUBLE
		case hkClassMember::TYPE_REAL:
			buffer[3] = ptr[0];
			buffer[2] = ptr[1];
			buffer[1] = ptr[2];
			buffer[0] = ptr[3];
			break;
#else
		case hkClassMember::TYPE_REAL:
#endif
		case hkClassMember::TYPE_INT64:
		case hkClassMember::TYPE_UINT64:
			buffer[7] = ptr[0];
			buffer[6] = ptr[1];
			buffer[5] = ptr[2];
			buffer[4] = ptr[3];
			buffer[3] = ptr[4];
			buffer[2] = ptr[5];
			buffer[1] = ptr[6];
			buffer[0] = ptr[7];
			break;
		case hkClassMember::TYPE_ULONG:
#if HK_POINTER_SIZE==4
			buffer[3] = ptr[0];
			buffer[2] = ptr[1];
			buffer[1] = ptr[2];
			buffer[0] = ptr[3];
#else /* HK_POINTER_SIZE==8 */
			buffer[7] = ptr[0];
			buffer[6] = ptr[1];
			buffer[5] = ptr[2];
			buffer[4] = ptr[3];
			buffer[3] = ptr[4];
			buffer[2] = ptr[5];
			buffer[1] = ptr[6];
			buffer[0] = ptr[7];
#endif
			break;

		case hkClassMember::TYPE_VECTOR4:
			_getEndianSwappedData( ptr, hkClassMember::TYPE_REAL, buffer);
			_getEndianSwappedData( ptr + sizeof(hkReal), hkClassMember::TYPE_REAL, buffer + sizeof(hkReal));
			_getEndianSwappedData( ptr + (2*sizeof(hkReal)), hkClassMember::TYPE_REAL, buffer + (2*sizeof(hkReal)));
			_getEndianSwappedData( ptr + (3*sizeof(hkReal)), hkClassMember::TYPE_REAL, buffer + (3*sizeof(hkReal)));
			break;
		default:
			HK_ASSERT(0x7fa195c4,0); // should not be able to get here
			break; 
	}
}

static int _getSimpleTypeSize(hkClassMember::Type t)
{
	switch (t)
	{
		case hkClassMember::TYPE_BOOL:
			return sizeof(hkBool);
		case hkClassMember::TYPE_CHAR:
			return sizeof(hkChar);
		case hkClassMember::TYPE_INT8:
		case hkClassMember::TYPE_UINT8:
			return sizeof(hkUint8);
		case hkClassMember::TYPE_INT16:
		case hkClassMember::TYPE_UINT16:
			return sizeof(hkUint16);
		case hkClassMember::TYPE_INT32:
		case hkClassMember::TYPE_UINT32:
			return sizeof(hkUint32);
		case hkClassMember::TYPE_INT64:
		case hkClassMember::TYPE_UINT64:
			return sizeof(hkUint64);
		case hkClassMember::TYPE_ULONG:
			return sizeof(hkUlong);
		case hkClassMember::TYPE_REAL:
			return sizeof(hkReal);
		case hkClassMember::TYPE_HALF:
			return sizeof(hkHalf);
		case hkClassMember::TYPE_VECTOR4:
			return sizeof(hkVector4);
			
	// should not be requested
	// as dealt with by sub type (eg matrix->set of vectors, quaternion == vector, etc)
		case hkClassMember::TYPE_ENUM:
		case hkClassMember::TYPE_QUATERNION:
		case hkClassMember::TYPE_MATRIX3:
		case hkClassMember::TYPE_ROTATION:
		case hkClassMember::TYPE_QSTRANSFORM:
		case hkClassMember::TYPE_MATRIX4:
		case hkClassMember::TYPE_TRANSFORM:
		case hkClassMember::TYPE_POINTER:
		case hkClassMember::TYPE_CSTRING:
		case hkClassMember::TYPE_STRINGPTR:
		case hkClassMember::TYPE_FUNCTIONPOINTER:
		case hkClassMember::TYPE_ARRAY:
		case hkClassMember::TYPE_SIMPLEARRAY:
		case hkClassMember::TYPE_HOMOGENEOUSARRAY:
		case hkClassMember::TYPE_VARIANT:
		case hkClassMember::TYPE_ZERO:
		case hkClassMember::TYPE_STRUCT:
		case hkClassMember::TYPE_INPLACEARRAY:
		case hkClassMember::TYPE_VOID:
		case hkClassMember::TYPE_MAX:
		default:
			{
				return 0;
			}
	}
}

void hkObjectSerializationManager::sendUpdatedMember(  hkUint64 id, const hkArray<hkInt16>& memberPath, hkClassMember::Type type, const void* ptr )
{	
	// don't write the packet size in a command to server.. historically like that..
	// We have it buffered so that we can not start to send command until we know we 
	// where able to interpret it and send it ok to that layout.
	hkDisplaySerializeOStream sendStream( m_streamWriter );
	sendStream.write8u(hkVisualDebuggerProtocol::HK_UPDATE_MEMBER);

	// IDs
	hkUint64 classID = getObjectType(id);
	sendStream.write64u( id ); // obj id
	sendStream.write64u( classID ); // class id

	// Member Path
	sendStream.write16u( (hkUint16) memberPath.getSize() );
	for (int mp=0; mp < memberPath.getSize(); ++mp)
	{
		sendStream.write16u( memberPath[mp] );
	}

	// Data packet, in the target platform form
	// The target will work out its own offest for the member (so we don't have
	// to worry about class optimizations etc. So that leaves endian and ptr size.
	// We don't allow ptr alterations at the mo, so just need to write to a proper 
	// endian altered stream.
	//XX In future we could compute the mem offest here instead on the server and just send
	// a byte offset instead of the member path. At least the member path can handle data
	// array resizes a bit better..

	hkUint32 sizeOfData = _getSimpleTypeSize(type);
	sendStream.write32u(sizeOfData);
	if (sizeOfData > 0)
	{
		if (m_targetLayout.getRules().m_littleEndian == hkStructureLayout::HostLayoutRules.m_littleEndian)
		{
			sendStream.writeRaw(ptr, sizeOfData); // has all the ids etc in it.
		}
		else // endian swap
		{
			char buffer[ sizeof(hkVector4) ]; // hkVector4 is the biggest thing we send across
			_getEndianSwappedData((char*)ptr, type, buffer);
			sendStream.writeRaw(buffer, sizeOfData);
		}
	}
}

bool hkObjectSerializationManager::isLive( hkUint64 ptrId )
{
	return m_liveIds.indexOf(ptrId) >= 0;
}

hkUint64 hkObjectSerializationManager::getID( const void* data )
{
	return m_dataReg.findObjectID( data );
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
