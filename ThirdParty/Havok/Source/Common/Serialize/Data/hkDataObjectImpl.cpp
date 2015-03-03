/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Data/hkDataObjectImpl.h>

#include <Common/Serialize/Data/hkDataObject.h>

//////////////////////////////////////////////////////////////////////////
// hkDataObjectImpl
//////////////////////////////////////////////////////////////////////////

/* static */void HK_CALL hkDataObjectImpl::assignValueImpl( hkDataObjectImpl* dstObj, MemberHandle handle, const Value& valueIn )
{
	hkDataObject::Type type = valueIn.getType();
	hkDataObject::Value dstVal(dstObj, handle);

	switch (type->getSubType())
	{
		case hkTypeManager::SUB_TYPE_POINTER:
		case hkTypeManager::SUB_TYPE_CLASS:
		{
			dstVal = valueIn.asObject();
			break;
		}
		case hkTypeManager::SUB_TYPE_BYTE:
		case hkTypeManager::SUB_TYPE_INT:
		{
			dstVal = valueIn.asInt();
			break;
		}
		case hkTypeManager::SUB_TYPE_REAL:
		{
			dstVal = valueIn.asReal();
			break;
		}
		case hkTypeManager::SUB_TYPE_TUPLE:
		{
			if (type->getParent()->isReal())
			{
				const int size = type->getTupleSize();
				dstVal.setVec(valueIn.asVec(size), size);
				break;
			}
			HK_ASSERT(0x324234, "Unhandled type");
			break;
		}
		case hkTypeManager::SUB_TYPE_CSTRING:
		{
			dstVal = valueIn.asString();
			break;
		}
		default:
		{
			HK_ASSERT(0x324234, "Unhandled type");
		}
	}
}


//////////////////////////////////////////////////////////////////////////
// hkDataArrayImpl
//////////////////////////////////////////////////////////////////////////

void hkDataArrayImpl::set( int index, const hkDataObject_Value& val )
{
	hkDataObject::Type type = getType();
	if( type->isTuple() && !type->getParent()->isReal())
	{
		HK_ASSERT(0x132f8fef, val.asArray().getSize() == hkDataArray(asArray(index)).getSize());
		setArray(index, val.asArray().getImplementation());
		return;
	}
	switch( getType()->getSubType() )
	{
		case hkTypeManager::SUB_TYPE_BYTE:
		case hkTypeManager::SUB_TYPE_INT:
			setInt(index, val.asInt());
			break;
		case hkTypeManager::SUB_TYPE_REAL:
			setReal(index, val.asReal());
			break;
		case hkTypeManager::SUB_TYPE_CLASS:
		case hkTypeManager::SUB_TYPE_POINTER:
			setObject( index, val.asObject().getImplementation() );
			break;
		case hkTypeManager::SUB_TYPE_CSTRING:
			setString( index, val.asString());
			break;
		case hkTypeManager::SUB_TYPE_ARRAY:
			setArray(index, val.asArray().getImplementation());
			break;
		case hkTypeManager::SUB_TYPE_TUPLE:
			if (type->getParent()->isReal())
			{
				setVec(index, val.asVec(type->getTupleSize()));
				break;
			}
			// Fall thru
		default:
			HK_ASSERT(0x77ca1cb5, 0);
	}
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
