/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/Util/hkSerializeDeprecated.h>
#include <Common/Compat/Deprecated/Packfile/Xml/hkXmlPackfileWriter.h>
#include <Common/Compat/Deprecated/Packfile/Xml/hkXmlPackfileReader.h>
#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>
#include <Common/Compat/Deprecated/Packfile/Binary/hkBinaryPackfileReader.h>
#include <Common/Serialize/Data/Dict/hkDataObjectDict.h>
#include <Common/Serialize/Data/Native/hkDataObjectNative.h>
#include <Common/Serialize/Data/Util/hkDataObjectUtil.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Base/Reflection/hkClassMemberAccessor.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Serialize/Util/hkClassPointerVtable.h>
#include <Common/Base/Config/hkProductFeatures.h>
#include <Common/Base/Reflection/Registry/hkClassNameRegistry.h>
#include <Common/Base/Config/hkOptionalComponent.h>

#define RAISE_ERROR(err, id, msg) if( err ) err->raiseError( id, msg )

static const char s_hkDataObjectTypeAttributeID[] = "hk.DataObjectType";

namespace // anonymous
{

struct TypeInfo
{
	enum
	{
		TYPE_INVALID,
		TYPE_BASIC,
		TYPE_ARRAY,
		TYPE_ENUM,
		TYPE_TUPLE,
		TYPE_POINTER,
		TYPE_CLASS,
		TYPE_VARIANT,
	};

	hkUint8 m_type;
	hkUint8 m_subType;
	hkUint8 m_tupleSize;
	hkUint8 _pad0;
};

#define HK_LUT(type, subType, tupleCount) \
{ hkUint8(TypeInfo::TYPE_##type), hkUint8(hkTypeManager::SUB_TYPE_##subType), hkUint8(tupleCount), 0 }

static const TypeInfo s_lut[] =
{
	HK_LUT(BASIC, VOID, 0),			//TYPE_VOID = 0,
	HK_LUT(BASIC, BYTE, 0),			//TYPE_BOOL,
	HK_LUT(BASIC, BYTE, 0),			//TYPE_CHAR,
	HK_LUT(BASIC, INT,	0),			//TYPE_INT8,
	HK_LUT(BASIC, BYTE, 0),			//TYPE_UINT8,
	HK_LUT(BASIC, INT,	0), 		//TYPE_INT16,
	HK_LUT(BASIC, INT,	0),			//TYPE_UINT16,
	HK_LUT(BASIC, INT,	0),			//TYPE_INT32,
	HK_LUT(BASIC, INT,  0),			//TYPE_UINT32,
	HK_LUT(BASIC, INT,  0),			//TYPE_INT64,
	HK_LUT(BASIC, INT,  0),			//TYPE_UINT64,
	HK_LUT(BASIC, REAL, 0),			//TYPE_REAL,

	HK_LUT(TUPLE, REAL, 4),			//TYPE_VECTOR4,
	HK_LUT(TUPLE, REAL, 4),			//TYPE_QUATERNION,
	HK_LUT(TUPLE, REAL,12),			//TYPE_MATRIX3,
	HK_LUT(TUPLE, REAL,12),			//TYPE_ROTATION,
	HK_LUT(TUPLE, REAL,12),			//TYPE_QSTRANSFORM,
	HK_LUT(TUPLE, REAL,16),			//TYPE_MATRIX4,
	HK_LUT(TUPLE, REAL,16),			//TYPE_TRANSFORM,

	HK_LUT(BASIC, VOID, 0),			//TYPE_ZERO,
	HK_LUT(POINTER, INVALID, 0),	//TYPE_POINTER,
	HK_LUT(BASIC, VOID, 0),			//TYPE_FUNCTIONPOINTER,
	HK_LUT(ARRAY, INVALID, 0),		//TYPE_ARRAY,
	HK_LUT(ARRAY, INVALID, 0),		//TYPE_INPLACEARRAY,
	HK_LUT(ENUM,  INVALID, 0),		//TYPE_ENUM,
	HK_LUT(CLASS, INVALID, 0),		//TYPE_STRUCT,
	HK_LUT(ARRAY, INVALID, 0),		//TYPE_SIMPLEARRAY,
	HK_LUT(ARRAY, INVALID, 0),		//TYPE_HOMOGENEOUSARRAY, //TODO: don't hardcode
	HK_LUT(VARIANT, INVALID, 0),		//TYPE_VARIANT,
	HK_LUT(BASIC, CSTRING, 0),		//TYPE_CSTRING,
	HK_LUT(BASIC, INT, 0),			//TYPE_ULONG,
	HK_LUT(ENUM, VOID, 0),			//TYPE_FLAGS,
	HK_LUT(BASIC, REAL, 0),			//TYPE_HALF,
	HK_LUT(BASIC, CSTRING, 0),		//TYPE_STRINGPTR,
	HK_LUT(ARRAY, INVALID, 0), 		//TYPE_RELARRAY
	HK_LUT(BASIC, VOID, 0),			//TYPE_MAX
};
HK_COMPILE_TIME_ASSERT(HK_COUNT_OF(s_lut) == (hkClassMember::TYPE_MAX + 1));

hkDataObject::Type HK_CALL getBasicType(hkTypeManager& typeManager, hkClassMember::Type type, const hkClass* klass)
{
	HK_ASSERT(0x56c8eb98, unsigned(type) < HK_COUNT_OF(s_lut));
	const TypeInfo& info = s_lut[type];

	switch (info.m_type)
	{
		default:
		{
			HK_ASSERT2(0x24424332, false, "Not a type that is basic");
			return HK_NULL;
		}
		case TypeInfo::TYPE_POINTER:
		{
			if (klass)
			{
				HK_ASSERT(0x432423a, klass);
				return typeManager.makePointer(typeManager.addClass(klass->getName()));
			}
			else
			{
				return typeManager.makePointer(typeManager.getHomogenousClass());
			}
		}
		case TypeInfo::TYPE_CLASS:
		{
			HK_ASSERT(0x432423a, klass);
			return typeManager.addClass(klass->getName());
		}
		case TypeInfo::TYPE_TUPLE:
		{
			// Its a tuple
			hkDataObject::Type base = typeManager.getSubType(hkTypeManager::SubType(info.m_subType));
			HK_ASSERT(0x243242a3, base);
			return typeManager.makeTuple(base, info.m_tupleSize);
		}
		case TypeInfo::TYPE_BASIC:
		{
			return typeManager.getSubType(hkTypeManager::SubType(info.m_subType));
		}
		case TypeInfo::TYPE_VARIANT:
		{
			return typeManager.makePointer(typeManager.getHomogenousClass());
		}
	}	
}

static hkBool HK_CALL _hasArrayMember(const hkClass& cls)
{
	for( int i = 0; i < cls.getNumMembers(); ++i )
	{
		switch( cls.getMember(i).getType() )
		{
			case hkClassMember::TYPE_ARRAY:
			case hkClassMember::TYPE_SIMPLEARRAY:
			case hkClassMember::TYPE_HOMOGENEOUSARRAY:
			{
				return true;
			}	
			default:
				break;
		}
	}
	return false;
}

static hkDataObject::Type HK_CALL legacyGetTypeFromMemberType(
	hkTypeManager& typeManager, hkClassMember::Type mtype, hkClassMember::Type stype, 
	const hkClass* klass, int count, const hkClassNameRegistry* nameReg, bool structToPointerInArrays)
{
	const TypeInfo& info = s_lut[mtype];

	hkDataObject::Type type = HK_NULL;

	switch (info.m_type)
	{
		case TypeInfo::TYPE_ENUM:
		{
			type = getBasicType(typeManager, stype, HK_NULL);
			break;
		}
		case TypeInfo::TYPE_CLASS:
		{
			// Struct
			type = getBasicType(typeManager, mtype, klass);
			break;
		}
		case TypeInfo::TYPE_POINTER:
		{
			type = typeManager.makePointer(getBasicType(typeManager, stype, klass));
			break;
		}
		case TypeInfo::TYPE_ARRAY:
		{
			// Its an array
			if (stype == hkClassMember::TYPE_STRUCT)
			{
				// If the subtype is a struct... perhaps I need to handle as an object
				HK_ASSERT(0x23423432, klass);
				type = typeManager.addClass(klass->getName());

				if(structToPointerInArrays && _hasArrayMember(*klass))
				{
  					// Make it an object type
  					type = typeManager.makePointer(type);
				}
			}
			else
			{
				type = getBasicType(typeManager, stype, klass);
			}

			// Tuple takes place before 'arrayizing'
			if( count != 0 )
			{
				type = typeManager.makeTuple(type, count);
			}

			// Its an array 
			return typeManager.makeArray(type);
		}
		case TypeInfo::TYPE_BASIC:
		{
			// Should be a basic type
			type = typeManager.getSubType(hkTypeManager::SubType(info.m_subType));
			break;
		}
		case TypeInfo::TYPE_TUPLE:
		{
			type = typeManager.makeTuple(typeManager.getSubType(hkTypeManager::SubType(info.m_subType)), info.m_tupleSize);
			break;
		}
		case TypeInfo::TYPE_VARIANT:
		{
			type = getBasicType(typeManager, mtype, klass);
			break;
		}
	}

	HK_ASSERT(0x2342423, type);
	if( count != 0 )
	{
		type = typeManager.makeTuple(type, count);
	}
	return type;
}

static hkDataClassImpl* HK_CALL legacyWrapClass(hkDataWorld* world, const hkClass& klass, const hkClassNameRegistry* nameReg, bool structToPointerInArrays)
{
	{
		hkDataClassImpl* cls = world->findClass(klass.getName());
		if (cls)
		{
			return cls;
		}
	}

	// Sort out the parent
	if (klass.getParent())
	{
		// Recusively set up the parent
		legacyWrapClass(world, *klass.getParent(), nameReg, structToPointerInArrays);
	}

	// Set up the info for the class
	hkDataClass::Cinfo info;
	info.name = klass.getName();
	info.parent = klass.getParent() ? klass.getParent()->getName() : HK_NULL;
	info.version = klass.getDescribedVersion();

	hkTypeManager& typeManager = world->getTypeManager();

	for( int i = 0; i < klass.getNumDeclaredMembers(); ++i )
	{
		const hkClassMember& m = klass.getDeclaredMember(i);
		hkDataObject::Type mtype = typeManager.getSubType(hkTypeManager::SUB_TYPE_VOID);
		
		if( !m.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) )
		{
			const hkClass* cls =  m.getClass();

			if( cls )
			{
				if( const hkVariant* typeAttr = m.getAttribute(s_hkDataObjectTypeAttributeID) )
				{
					const hkClassMemberAccessor attrTypeName(*typeAttr, "typeName");
					cls = nameReg->getClassByName(attrTypeName.asCstring());
				}
				else if( const hkVariant* classAttr = cls->getAttribute(s_hkDataObjectTypeAttributeID) )
				{
					const hkClassMemberAccessor attrTypeName(*classAttr, "typeName");
					cls = nameReg->getClassByName(attrTypeName.asCstring());
				}

				// We should still have a class
				HK_ASSERT(0x324323bb, cls);
			}

			mtype = legacyGetTypeFromMemberType(typeManager, m.getType(), m.getSubType(), cls, m.getCstyleArraySize(), nameReg, structToPointerInArrays);	
		}

		// Add the member (even if not serialized, will have type VOID)
		hkDataClass::Cinfo::Member& dstMem = info.members.expandOne();

		dstMem.name = m.getName();
		dstMem.valuePtr = HK_NULL;
		dstMem.type = mtype;
	}
	return world->newClass(info);
}

class ClassWrapper: public hkVersionPatchManager::ClassWrapper
{
public:
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);

	virtual hkDataClassImpl* wrapClass(hkDataWorld* world, const char* typeName)
	{
		hkDataClassImpl* impl = world->findClass(typeName);
		if (impl)
		{
			return impl;
		}

		const hkClass* k = m_nameReg->getClassByName(typeName); // present
		if (!k)
		{
			return HK_NULL;
		}

		return legacyWrapClass(world, *k, m_nameReg, m_useStructToPointerInArrays);
	}

	ClassWrapper(bool useStructToPointerInArrays, const hkClassNameRegistry* nameReg = HK_NULL)
		: m_useStructToPointerInArrays(useStructToPointerInArrays)
	{
		if (!nameReg)
		{
			nameReg = hkBuiltinTypeRegistry::getInstance().getClassNameRegistry();
		}
		HK_ASSERT(0x324324b4, nameReg);

		m_nameReg = nameReg;
	}

	hkRefPtr<const hkClassNameRegistry> m_nameReg;
	bool m_useStructToPointerInArrays;
};

class DataWorldNative: public hkDataWorldNative
{
public:
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);

	DataWorldNative(bool useStructToPointerInArrays) : m_useStructToPointerInArrays(useStructToPointerInArrays) 
	{}
	
	virtual hkDataObject::Type getTypeFromMemberType(hkClassMember::Type mtype, hkClassMember::Type stype, const hkClass* klass, int count)
	{
		return legacyGetTypeFromMemberType(m_typeManager, mtype, stype, klass, count, m_reg, m_useStructToPointerInArrays);
	}

protected:
	bool m_useStructToPointerInArrays;
};


} // namespace anonymous

struct hkSerializeDeprecated2 : public hkSerializeDeprecated
{
	hkResult saveXmlPackfile( const void* object, const hkClass& klass, hkStreamWriter* stream, const hkPackfileWriter::Options& options, hkPackfileWriter::AddObjectListener* userListener, hkSerializeUtil::ErrorDetails* errorOut ) HK_OVERRIDE
	{
		HK_OPTIONAL_COMPONENT_MARK_USED(hkSerializeDeprecated);
		hkXmlPackfileWriter writer;
		writer.setContents(object, klass, userListener);
		return writer.save(stream, options);
	}

	hkBool32 isLoadable(const hkSerializeUtil::FormatDetails& details) HK_OVERRIDE
	{
		HK_OPTIONAL_COMPONENT_MARK_USED(hkSerializeDeprecated);
		switch( details.m_formatType )
		{
			case hkSerializeUtil::FORMAT_PACKFILE_BINARY:
				return hkString::memCmp( &details.m_layoutRules, &hkStructureLayout::HostLayoutRules, sizeof(hkStructureLayout::HostLayoutRules)) == 0;
			case hkSerializeUtil::FORMAT_PACKFILE_XML:
				return true;
			case hkSerializeUtil::FORMAT_TAGFILE_BINARY: // deprecated lib can't load it
			case hkSerializeUtil::FORMAT_ERROR:
			case hkSerializeUtil::FORMAT_UNKNOWN:
			default:
				return false;
		}
	}

	static hkResult packfileToDictWorld( hkDataWorldDict& dictWorld, hkPackfileReader& reader, hkStreamReader& sr, const hkSerializeUtil::FormatDetails& details, hkSerializeUtil::ErrorDetails* errorOut )
	{
		if( reader.loadEntireFile( &sr ) != HK_SUCCESS )
		{
			RAISE_ERROR( errorOut, hkSerializeUtil::ErrorDetails::ERRORID_LOAD_FAILED, "Failed to load file" );
			return HK_FAILURE;
		}

		if( hkString::strCmp( reader.getContentsVersion(), "Havok-7.0.0-r1" ) <= 0 )
		{
			hkRefPtr<hkVersionRegistry> userVersionReg = &hkVersionRegistry::getInstance();
			if( hkVersionUtil::updateToVersion(reader, *userVersionReg, "Havok-7.0.0-r1") != HK_SUCCESS )
			{
				RAISE_ERROR( errorOut, hkSerializeUtil::ErrorDetails::ERRORID_VERSIONING_FAILED, "Unable to version contents. Your assets might be older than HK_SERIALIZE_MIN_COMPATIBLE_VERSION (if defined).");
				return HK_FAILURE;
			}
		}

		hkBool32 useStructToPointerInArrays = false;
		if( hkString::strCmp(reader.getContentsVersion(), "hk_2011.1.0-r1" ) < 0)
		{
			// Before 2011.1, the old serialization worked by creating arrays of pointers
			// from arrays of objects. Starting from 2011.1 we have that this is no longer
			// necessary.
			useStructToPointerInArrays = true;
		}

		{
			//reader.getContents(HK_NULL);
			const hkClassNameRegistry* classReg = reader.getClassNameRegistry();

			if(!classReg)
			{
				RAISE_ERROR( errorOut, hkSerializeUtil::ErrorDetails::ERRORID_VERSIONING_FAILED, "File contents are from an unknown SDK version");
				return HK_FAILURE;
			}

			const hkArray<hkVariant>& loadedObjects = reader.getLoadedObjects();
			hkStringMap<const hkClass*> classFromName;

			// This next bit is nasty: some objects have struct members which have vtables.
			// normally the finish ctor will will fix these up for us, but in this case we're faking
			// the finish ctor so all the vtables are null. We have to fix them up ourselves

// 			//hack to find all classes which have virtual members by value - they need their vtables restored on load
// 			{
// 				hkArray<const hkClass*> classes;
// 				hkVersionRegistry::getInstance().getClassNameRegistry(HK_COMPAT_VERSION_TO::VersionString)->getClasses(classes);
// 				for( int ci = 0; ci < classes.getSize(); ++ci )
// 				{
// 					const hkClass& k = *classes[ci];
// 					for( int mi = 0; mi < k.getNumDeclaredMembers(); ++mi )
// 					{
// 						const hkClassMember& m = k.getDeclaredMember(mi);
// 						if( m.hasClass() && m.getType() == hkClassMember::TYPE_STRUCT && m.getClass()->hasVtable() )
// 						{
// 							printf("{ %s, %s }, // -> %s\n", k.getName(), m.getName(), m.getClass()->getName() );
// 						}
// 					}
// 				}
// 			}

			for( int li = 0; li < loadedObjects.getSize(); ++li )
			{
				const hkVariant& obj = loadedObjects[li];
				classFromName.insert( obj.m_class->getName(), loadedObjects[li].m_class );
				static const char* const classesWithEmbeddedVtables[][2] = 
				{
					{ "hkpVehicleLinearCastWheelCollide", "rejectChassisListener" }, // -> hkpRejectChassisListener
					{ "hkpConvexTransformShapeBase", "childShape" }, // -> hkpSingleShapeContainer
					{ "hkpTriSampledHeightFieldBvTreeShape", "childContainer" }, // -> hkpSingleShapeContainer
					{ "hkaiWorld", "streamingManager" }, // -> hkaiNavMeshManager
					{ "hkaiWorld", "cutter" }, // -> hkaiNavMeshCutter
					{ "hkbSenseHandleModifier", "handle" }, // -> hkbHandle
					{ "hkpBvShape", "childShape" }, // -> hkpSingleShapeContainer
					{ "hkpVehicleRayCastWheelCollide", "rejectRayChassisListener" }, // -> hkpRejectChassisListener
					{ "hkpMoppBvTreeShape", "child" }, // -> hkpSingleShapeContainer
					{ "hkbEvaluateHandleModifier", "oldHandle" }, // -> hkbHandle
					{ "hkpTransformShape", "childShape" }, // -> hkpSingleShapeContainer
					//"hkpEntity" "motion" -> hkpMaxSizeMotion // special case below because it is so weird
				};
				for( int i = 0; i < (int)HK_COUNT_OF(classesWithEmbeddedVtables); ++i )
				{
					const hkClass* ok = classReg->getClassByName(classesWithEmbeddedVtables[i][0]);
					if( ok && ok->isSuperClass(*obj.m_class) )
					{
						hkClassMemberAccessor member( obj.m_object, *obj.m_class, classesWithEmbeddedVtables[i][1]);
						const hkClass* mk = member.getClassMember().getClass();
						HK_ASSERT(0x678ac2dd, mk);
						*(const void**)member.getAddress() = mk;
					}
				}
				{
					const hkClass* ok = classReg->getClassByName("hkpEntity");
					if( ok && ok->isSuperClass(*obj.m_class) )
					{
						hkClassMemberAccessor motion( obj.m_object, *obj.m_class, "motion");
						int mtype = motion.member("type").asInt8();
						enum MotionType
						{
							MOTION_INVALID,
							MOTION_DYNAMIC,
							MOTION_SPHERE_INERTIA,
							MOTION_BOX_INERTIA,
							MOTION_KEYFRAMED,
							MOTION_FIXED,
							MOTION_THIN_BOX_INERTIA,
							MOTION_CHARACTER,
							MOTION_MAX_ID
						};
						static const char* classFromMotionType[] =
						{
							HK_NULL,
							"hkpBoxMotion",
							"hkpSphereMotion",
							"hkpBoxMotion",
							"hkpKeyframedRigidMotion",
							"hkpFixedRigidMotion",
							"hkpThinBoxMotion",
							"hkpCharacterMotion",
							HK_NULL
						};
						// motion is actually a union - not a max size motion
						HK_ASSERT(0x2025471e, unsigned(mtype) < HK_COUNT_OF(classFromMotionType) );
						const hkClass* mk = classReg->getClassByName(classFromMotionType[mtype]);
						HK_ASSERT(0x26379846, mk);
						*(const void**)motion.getAddress() = mk;
					}
				}
			}
			hkClassPointerVtable::TypeInfoRegistry finishReg(classFromName);
			hkClassPointerVtable::VtableRegistry vtableReg;
			void* contents = reader.getContentsWithRegistry(HK_NULL, &finishReg);
			reader.getPackfileData()->disableDestructors();
			DataWorldNative nativeWorld(useStructToPointerInArrays);
			nativeWorld.setClassRegistry(classReg);
			nativeWorld.setVtableRegistry(&vtableReg);
			const hkClass* contentsClass = classFromName.getWithDefault( reader.getContentsClassName(), HK_NULL );
			HK_ASSERT(0x71c8a410, contentsClass);
			nativeWorld.setContents( contents, *contentsClass );

			hkDataObjectUtil::deepCopyWorld( dictWorld, nativeWorld );
			
			//#include <Common/Serialize/Tagfile/Text/hkTextTagfileWriter.h>
			//hkTextTagfileWriter ttd; ttd.save(dictWorld.getContents(), hkOstream("xxxd.txt").getStreamWriter(), HK_NULL);
			//hkTextTagfileWriter ttn; ttn.save(nativeWorld.getContents(), hkOstream("xxxn.txt").getStreamWriter(), HK_NULL);

			hkVersionPatchManager::ClassWrapper* wrapper = HK_NULL;
			if (details.m_formatVersion < 9)
			{
				// Legacy
				wrapper = new ClassWrapper(useStructToPointerInArrays);
			}
			else 
			{
				// Current
				wrapper = new hkDefaultClassWrapper;
			}
			
			hkResult res = hkVersionPatchManager::getInstance().applyPatches(dictWorld, wrapper);
			wrapper->removeReference();

			if( res != HK_SUCCESS )
			{
				RAISE_ERROR( errorOut, hkSerializeUtil::ErrorDetails::ERRORID_VERSIONING_FAILED, "Unable to version contents, check warning log");
			}
			return res;
		}
	}

	hkResult packfileToDictWorld(hkDataWorldDict& dictWorld, hkStreamReader& sr, const hkSerializeUtil::FormatDetails& details, hkSerializeUtil::ErrorDetails* errorOut)
	{
		HK_OPTIONAL_COMPONENT_MARK_USED(hkSerializeDeprecated);
		switch( details.m_formatType )
		{
			case hkSerializeUtil::FORMAT_PACKFILE_BINARY:
			{
				hkBinaryPackfileReader reader;
				return packfileToDictWorld(dictWorld, reader, sr, details, errorOut);
			}
			case hkSerializeUtil::FORMAT_PACKFILE_XML:
			{
				hkXmlPackfileReader reader;
				return packfileToDictWorld(dictWorld, reader, sr, details, errorOut);
			}
			default:
			{
				return HK_FAILURE;
			}
		}
	}

	hkResource* loadOldPackfile(hkStreamReader& sr, const hkSerializeUtil::FormatDetails& details, hkSerializeUtil::ErrorDetails* errorOut) HK_OVERRIDE
	{
		HK_OPTIONAL_COMPONENT_MARK_USED(hkSerializeDeprecated);
		hkDataWorldDict world;
		if( packfileToDictWorld(world, sr, details, errorOut ) == HK_SUCCESS )
		{
			return hkDataObjectUtil::toResource(world.getContents(), true);
		}
		return HK_NULL;
	}

	hkObjectResource* loadOldPackfileOnHeap(hkStreamReader& sr, const hkSerializeUtil::FormatDetails& details, hkSerializeUtil::ErrorDetails* errorOut) HK_OVERRIDE
	{
		HK_OPTIONAL_COMPONENT_MARK_USED(hkSerializeDeprecated);
		hkDataWorldDict world;
		if( packfileToDictWorld(world, sr, details, errorOut ) == HK_SUCCESS )
		{
			return hkDataObjectUtil::toObject(world.getContents(), true);
		}
		return HK_NULL;
	}


	virtual hkResult readXmlPackfileHeader(hkStreamReader* stream, XmlPackfileHeader& out, hkSerializeUtil::ErrorDetails* errorOut) HK_OVERRIDE
	{
		hkXmlPackfileReader reader;
		return reader.readHeader(stream, out);
	}

	virtual bool isEnabled() const HK_OVERRIDE
	{
		return true;
	}
};

void HK_CALL hkSerializeDeprecated::initDeprecated()
{
	static hkSerializeDeprecated2 s_instance2;
	// Prevent this instance from being deleted
	s_instance2.addReference();
	hkSerializeDeprecated::replaceInstance(&s_instance2);
}
HK_OPTIONAL_COMPONENT_DEFINE_MANUAL(hkSerializeDeprecated, hkSerializeDeprecated::initDeprecated);

void HK_CALL hkFeature_serializeDeprecatedPre700()
{
	hkSerializeDeprecated::initDeprecated();
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
