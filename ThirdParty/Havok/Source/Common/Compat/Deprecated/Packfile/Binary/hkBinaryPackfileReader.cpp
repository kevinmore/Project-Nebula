/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Compat/Deprecated/Packfile/Binary/hkBinaryPackfileReader.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Base/Reflection/hkClassMemberAccessor.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Serialize/Copier/hkObjectCopier.h>
#include <Common/Serialize/Packfile/Binary/hkPackfileHeader.h>
#include <Common/Serialize/Packfile/Binary/hkPackfileSectionHeader.h>
#include <Common/Serialize/Util/hkChainedClassNameRegistry.h>
#include <Common/Serialize/Util/hkObjectInspector.h>
#include <Common/Serialize/Util/hkStaticClassNameRegistry.h>
#include <Common/Compat/Deprecated/Version/hkPackfileObjectUpdateTracker.h>
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Base/Config/hkConfigVersion.h> // for HAVOK_BUILD_NUMBER

#if 0
static int tab = 0;
#	define REPORT(INDENT, TEXT) { \
		char reportBuf[512]; \
		hkErrStream ostr(reportBuf,sizeof(reportBuf)); \
		int indent = INDENT; while(indent-- > 0){ostr << "\t";}; \
		ostr << TEXT; \
		hkError::getInstance().message(hkError::MESSAGE_REPORT, 0, reportBuf, "Binary", 0); \
	}

#	define TRACE_FUNC_ENTER(A) REPORT(tab++, A)
#	define TRACE_FUNC_EXIT(A) REPORT(--tab, A)
#	define TRACE(A) REPORT(tab, A)
#else
#	define TRACE_FUNC_ENTER(A)
#	define TRACE_FUNC_EXIT(A)
#	define TRACE(A)
#endif

extern const hkClass hkClassVersion1Class;
extern const hkClass hkClassVersion1PaddedClass;
extern const hkClass hkClassMemberVersion1Class;
extern const hkClass hkClassMemberVersion3Class;
extern const hkClass hkClassVersion3Class;
namespace
{
	typedef hkPointerMap<const void*, const hkClass*> ClassFromAddressMap;
	struct Location
	{
		int m_sectionIndex;
		int m_offset;
	};

	struct DataEntry
	{
		int m_sectionIndex;
		int m_offset;
		int m_classIndex;
	};

	static void setVariantClassPointers( hkVariant* v, int numVariants, const ClassFromAddressMap& classFromObject )
	{
		for( int i = 0; i < numVariants; ++i )
		{
			if( v[i].m_object && v[i].m_class == HK_NULL )
			{
				v[i].m_class = classFromObject.getWithDefault(v[i].m_object, HK_NULL);
				if( v[i].m_class == HK_NULL )
				{
					HK_WARN_ALWAYS(0x067fde46, "Can not find class pointer for an object at 0x" << v[i].m_object << ".\n"
						<< "You will have to set manually corresponding class pointer in the variant. Otherwise you have to store metadata in the packfile.");
				}
			}
		}
	}

	static void resurrectVariantClassPointers( void* pointer, const hkClass& klass, const ClassFromAddressMap& classFromObject, int numObjs )
	{
		for( int memberIndex = 0; memberIndex < klass.getNumMembers(); ++memberIndex )
		{
			const hkClassMember& member = klass.getMember(memberIndex);
			if( member.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) )
			{
				continue;
			}
			switch( member.getType() )
			{
				case hkClassMember::TYPE_VARIANT:
				{
					int nelem = member.getCstyleArraySize() ? member.getCstyleArraySize() : 1;
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						setVariantClassPointers( static_cast<hkVariant*>(maccess.asRaw()), nelem, classFromObject );
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
					break;
				}
				case hkClassMember::TYPE_ARRAY:
				case hkClassMember::TYPE_SIMPLEARRAY:
				{
					if( member.getSubType() == hkClassMember::TYPE_VARIANT )
					{
						void* obj = pointer;
						int objCount = numObjs;
						while( --objCount >= 0 )
						{
							hkClassMemberAccessor maccess(obj, &member);
							hkClassMemberAccessor::SimpleArray& array = maccess.asSimpleArray();
							setVariantClassPointers( static_cast<hkVariant*>(array.data), array.size, classFromObject );
							obj = hkAddByteOffset(obj, klass.getObjectSize());
						}
					}
					else if( member.getSubType() == hkClassMember::TYPE_STRUCT )
					{
						HK_ASSERT(0x4b220a34, member.hasClass());
						void* obj = pointer;
						int objCount = numObjs;
						while( --objCount >= 0 )
						{
							hkClassMemberAccessor maccess(obj, &member);
							hkClassMemberAccessor::SimpleArray& array = maccess.asSimpleArray();
							resurrectVariantClassPointers( array.data, member.getStructClass(), classFromObject, array.size );
							obj = hkAddByteOffset(obj, klass.getObjectSize());
						}
					}
					break;
				}
				case hkClassMember::TYPE_STRUCT:
				{
					HK_ASSERT(0x339f5834, member.hasClass());
					int nelem = member.getCstyleArraySize() ? member.getCstyleArraySize() : 1;
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						resurrectVariantClassPointers( maccess.asRaw(), member.getStructClass(), classFromObject, nelem );
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
					break;
				}
				case hkClassMember::TYPE_HOMOGENEOUSARRAY:
				{
					HK_ASSERT(0x4a4b6a18, member.getCstyleArraySize() == 0);
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						hkClassMemberAccessor::HomogeneousArray& array = maccess.asHomogeneousArray();
						if( array.data && array.klass )
						{
							resurrectVariantClassPointers( array.data, *array.klass, classFromObject, array.size );
						}
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
					break;
				}
				default:
				{
					// skip over all other types
				}
			}
		}
	}
		// Fills out m_collectedObjects - used for finding all objects in a packfile.
	class PackfileObjectsCollector: public hkObjectInspector::ObjectListener
	{
		public:
			PackfileObjectsCollector( hkArray<hkVariant>& collectedObjects, const ClassFromAddressMap& classesForVObjects,
				int fileVersion, const char* contentsVersion, hkPackfileReader::UpdateFlagFromClassMap& updateFlagFromClass )
				: m_collectedObjects(collectedObjects), m_classesForVObjects(classesForVObjects),
				 m_fileVersion(fileVersion), m_contentsVersion(contentsVersion), m_updateFlagFromClass(updateFlagFromClass) { }
			virtual hkResult objectCallback( const void* objP, const hkClass& klass, hkArray<hkObjectInspector::Pointer>::Temp& containedPointers );

		private:

			hkArray<hkVariant>& m_collectedObjects; // we fill in this array
			const ClassFromAddressMap& m_classesForVObjects; // exact type of virtual objects
			enum Flags { NOT_SEEN=0, PENDING=1, DONE=2 };
			hkPointerMap<const void*, Flags> m_seenObjects;
			int m_fileVersion;
			const char* m_contentsVersion;
			hkPackfileReader::UpdateFlagFromClassMap& m_updateFlagFromClass;
	};

	hkResult PackfileObjectsCollector::objectCallback( const void* objP, const hkClass& klass, hkArray<hkObjectInspector::Pointer>::Temp& containedPointers )
	{
		HK_ASSERT(0x6b56a08d, m_seenObjects.getWithDefault(objP,PackfileObjectsCollector::Flags(0)) < PackfileObjectsCollector::DONE );
		{
			HK_SERIALIZE_LOG(("CollectObject(addr=0x%p,klass=\"%s\")\n", objP, klass.getName()));
			hkVariant& v = m_collectedObjects.expandOne();
			v.m_class = &klass;
			v.m_object = const_cast<void*>(objP);
			m_seenObjects.insert(objP, PackfileObjectsCollector::DONE);
			hkArray<hkObjectInspector::Pointer>::Temp pendingPointers;
			pendingPointers.reserveExactly(containedPointers.getSize());
			for( int ptrIdx = 0; ptrIdx < containedPointers.getSize(); ++ptrIdx )
			{
				hkObjectInspector::Pointer& p = containedPointers[ptrIdx];
				void* objPtr = *(p.location);
				// update the klass for virtual objects
				// the klass must be correct for farther processing
				HK_ON_DEBUG( if( objPtr ) HK_SERIALIZE_LOG(("CollectObjectHasPointer(addr=0x%p,klass=\"%s\",klasssig=0x%x,seen=%i)\n", objPtr, p.klass->getName(), p.klass->getSignature(), m_seenObjects.hasKey(objPtr)?1:0)) );
				if( objPtr != HK_NULL && m_seenObjects.hasKey(objPtr) == false )
				{
					p.klass = m_classesForVObjects.getWithDefault(objPtr, p.klass);
					m_seenObjects.insert(objPtr, PackfileObjectsCollector::PENDING );
					HK_ASSERT2(0x15a09058, p.klass != HK_NULL, "Cannot walk through an object of unknown type. Are you trying to version old packfile with no meta-data?");
					hkPackfileReader::updateMetaDataInplace( const_cast<hkClass*>(p.klass), m_fileVersion, m_contentsVersion, m_updateFlagFromClass );
					//PRINT(("PTR\t0x%p\t# %s(0x%x) at 0x%p\n", objPtr, p.klass->getName(), p.klass->getSignature(), p.klass));
					pendingPointers.expandOne() = p;
				}
			}
			containedPointers = pendingPointers;
			// resurrect hkVariant class pointers
			resurrectVariantClassPointers(const_cast<void*>(objP), klass, m_classesForVObjects, 1);
		}
		return HK_SUCCESS;
	}

	// Build up a map of global pointers
	class PackfilePointersMapListener: public hkObjectInspector::ObjectListener
	{
		public:
			PackfilePointersMapListener(hkBinaryPackfileUpdateTracker& tracker, const hkClassNameRegistry& classReg,
				int fileVersion, const char* contentsVersion, hkPackfileReader::UpdateFlagFromClassMap& updateFlagFromClass)
				: m_tracker(tracker), m_classReg(classReg), m_fileVersion(fileVersion), m_contentsVersion(contentsVersion), m_updateFlagFromClass(updateFlagFromClass) { }
			virtual hkResult objectCallback( const void* objP, const hkClass& klass, hkArray<hkObjectInspector::Pointer>::Temp& containedPointers );

		private:
			void updateClassPointers(hkObjectInspector::Pointer& p);

			hkBinaryPackfileUpdateTracker& m_tracker; // we fill in this array
			const hkClassNameRegistry& m_classReg; // class registry to find exact type of virtual objects
			int m_fileVersion;
			const char* m_contentsVersion;
			hkPackfileReader::UpdateFlagFromClassMap& m_updateFlagFromClass;
			enum Flags { NOT_SEEN=0, PENDING=1, DONE=2 };
			hkPointerMap<const void*, Flags> m_seenObjects;
	};

	void PackfilePointersMapListener::updateClassPointers(hkObjectInspector::Pointer& p)
	{
		HK_ASSERT2(0x15a09058, p.klass != HK_NULL, "Cannot walk through an object of unknown type. Are you trying to version old packfile with no meta-data?");
		void* objPtr = *(p.location);
		HK_ASSERT(0x5cf1220f, objPtr);
		hkPackfileReader::updateMetaDataInplace( const_cast<hkClass*>(p.klass), m_fileVersion, m_contentsVersion, m_updateFlagFromClass );
		if( p.klass->hasVtable() )
		{
			// find exact class for the object
			// NOTE: The tracker.m_finish must have all the virtual object infos.
			const char* className = m_tracker.m_finish.getWithDefault(objPtr, 0);
			HK_ASSERT3(0x15a0905c, className != HK_NULL, "The virtual object at 0x" << objPtr << " is not found in the finish table, base class is " << p.klass->getName());
			p.klass = m_classReg.getClassByName(className);
			HK_ASSERT3(0x15a0905d, p.klass != HK_NULL, "The class " << className << " is not registered?");
			hkPackfileReader::updateMetaDataInplace( const_cast<hkClass*>(p.klass), m_fileVersion, m_contentsVersion, m_updateFlagFromClass );
		}
	}

	hkResult PackfilePointersMapListener::objectCallback( const void* objP, const hkClass& klass, hkArray<hkObjectInspector::Pointer>::Temp& containedPointers )
	{
		HK_ASSERT(0x7cc342ab, m_seenObjects.getWithDefault(objP,Flags(0)) < DONE );
		m_seenObjects.insert(objP, DONE);
		hkArray<hkObjectInspector::Pointer>::Temp pendingPointers;
		pendingPointers.reserveExactly(containedPointers.getSize());
		//PRINT(("WALK\t0x%p\t# %s(0x%x) at 0x%p\n", objP, klass.getName(), klass.getSignature(), &klass));
		for( int ptrIdx = 0; ptrIdx < containedPointers.getSize(); ++ptrIdx )
		{
			hkObjectInspector::Pointer& p = containedPointers[ptrIdx];
			void* objPtr = *(p.location);
			HK_SERIALIZE_LOG(("ObjectPointedBy(obj=0x%p,loc=0x%p)\n", objPtr, p.location));
			m_tracker.objectPointedBy(objPtr, p.location);
			//PRINT(("GPTR\t0x%p\tat\t0x%p\t# %s(0x%x) at 0x%p\n", objPtr, p.location, p.klass->getName(), p.klass->getSignature(), p.klass));
			if( objPtr != HK_NULL && m_seenObjects.hasKey(objPtr) == false )
			{
				m_seenObjects.insert(objPtr, PENDING);
				updateClassPointers(p);
				pendingPointers.expandOne() = p;
			}
		}
		containedPointers.swap(pendingPointers);
		return HK_SUCCESS;
	}

	// Check alignment of c-strings and reallocate when required, must be at least 2-byte aligned to support hkStringPtr
	class PackfileCstringListener: public hkObjectInspector::ObjectListener
	{
		public:
			PackfileCstringListener(hkBinaryPackfileUpdateTracker& tracker, const hkClassNameRegistry& classReg)
				: m_tracker(tracker), m_classReg(classReg) { }
			virtual hkResult objectCallback( const void* objP, const hkClass& klass, hkArray<hkObjectInspector::Pointer>::Temp& containedPointers );

		private:
			hkBinaryPackfileUpdateTracker& m_tracker; // we fill in this array
			const hkClassNameRegistry& m_classReg; // class registry to find exact type of virtual objects
			enum Flags { NOT_SEEN=0, PENDING=1, DONE=2 };
			hkPointerMap<const void*, Flags> m_seenObjects;

			void updateObjectCstrings(void* object, const hkClass& klass, int numObjs);
			void checkAndReallocateCstrings(char** strings, int nelem);
	};

	hkResult PackfileCstringListener::objectCallback( const void* objP, const hkClass& klass, hkArray<hkObjectInspector::Pointer>::Temp& containedPointers )
	{
		HK_ASSERT(0x7cc342ab, m_seenObjects.getWithDefault(objP, Flags(0)) < DONE );
		m_seenObjects.insert(objP, DONE);
		hkArray<hkObjectInspector::Pointer>::Temp pendingPointers;
		pendingPointers.reserveExactly(containedPointers.getSize());
		for( int ptrIdx = 0; ptrIdx < containedPointers.getSize(); ++ptrIdx )
		{
			hkObjectInspector::Pointer& p = containedPointers[ptrIdx];
			void* objPtr = *(p.location);
			if( objPtr != HK_NULL && m_seenObjects.hasKey(objPtr) == false )
			{
				m_seenObjects.insert(objPtr, PENDING);
				pendingPointers.expandOne() = p;
			}
		}
		containedPointers.swap(pendingPointers);
		updateObjectCstrings(const_cast<void*>(objP), klass, 1);
		return HK_SUCCESS;
	}

	void PackfileCstringListener::checkAndReallocateCstrings(char** strings, int nelem)
	{
		for( int i = 0; i < nelem; ++i )
		{
			if( hkUlong(strings[i]) & hkStringPtr::OWNED_FLAG )
			{
				strings[i] = hkString::strDup(strings[i]); // reallocate, 16-byte aligned by default
				m_tracker.addAllocation(strings[i]);
			}
		}
	}

	void PackfileCstringListener::updateObjectCstrings(void* pointer, const hkClass& klass, int numObjs)
	{
		for( int memberIndex = 0; memberIndex < klass.getNumMembers(); ++memberIndex )
		{
			const hkClassMember& member = klass.getMember(memberIndex);
			if( member.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) )
			{
				continue;
			}
			switch( member.getType() )
			{
				case hkClassMember::TYPE_CSTRING:
				{
					int nelem = member.getCstyleArraySize() ? member.getCstyleArraySize() : 1;
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						checkAndReallocateCstrings( static_cast<char**>(maccess.asRaw()), nelem );
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
					break;
				}
				case hkClassMember::TYPE_ARRAY:
				case hkClassMember::TYPE_SIMPLEARRAY:
				{
					if( member.getSubType() == hkClassMember::TYPE_STRUCT )
					{
						HK_ASSERT(0x4b220a34, member.hasClass());
						void* obj = pointer;
						int objCount = numObjs;
						while( --objCount >= 0 )
						{
							hkClassMemberAccessor maccess(obj, &member);
							hkClassMemberAccessor::SimpleArray& array = maccess.asSimpleArray();
							updateObjectCstrings( array.data, member.getStructClass(), array.size );
							obj = hkAddByteOffset(obj, klass.getObjectSize());
						}
					}
					else if( member.getSubType() == hkClassMember::TYPE_CSTRING )
					{
						void* obj = pointer;
						int objCount = numObjs;
						while( --objCount >= 0 )
						{
							hkClassMemberAccessor maccess(obj, &member);
							hkClassMemberAccessor::SimpleArray& array = maccess.asSimpleArray();
							checkAndReallocateCstrings( static_cast<char**>(array.data), array.size );
							obj = hkAddByteOffset(obj, klass.getObjectSize());
						}
					}
					break;
				}
				case hkClassMember::TYPE_STRUCT:
				{
					HK_ASSERT(0x339f5834, member.hasClass());
					int nelem = member.getCstyleArraySize() ? member.getCstyleArraySize() : 1;
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						updateObjectCstrings( maccess.asRaw(), member.getStructClass(), nelem );
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
					break;
				}
				case hkClassMember::TYPE_HOMOGENEOUSARRAY:
				{
					HK_ASSERT(0x4a4b6a18, member.getCstyleArraySize() == 0);
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						hkClassMemberAccessor::HomogeneousArray& array = maccess.asHomogeneousArray();
						if( array.data && array.klass )
						{
							updateObjectCstrings( array.data, *array.klass, array.size );
						}
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
					break;
				}
				default:
				{
					// skip over all other types
				}
			}
		}
	}

    static int findEndFixupIndex( int* fixupsStart, int byteSize, int numIntsPerFixup )
    {
	    if( byteSize == 0 )
	    {
		    return 0;
	    }
	    HK_ASSERT(0x770f62e2, (byteSize==0) || (byteSize >= numIntsPerFixup*hkSizeOf(hkInt32)) );
	    int numIntsPad16 = byteSize / hkSizeOf(hkInt32);
	    int curIndex = numIntsPad16 - (numIntsPad16 % numIntsPerFixup);
	    while( 1 )
	    {
		    if( fixupsStart[curIndex] >= 0 )
		    {
			    return curIndex + numIntsPerFixup;
		    }
		    else
		    {
			    HK_ASSERT(0x770f62e2, curIndex >= 0 );
			    curIndex -= numIntsPerFixup;
		    }
	    }
    }

    const char* getOriginalContentsVersion(const hkPackfileHeader& header)
    {
	    if( header.m_contentsVersion[0] != char(-1) )
	    {
		    return header.m_contentsVersion;
	    }
	    else if( header.m_fileVersion == 1 )
	    {
		    return "Havok-3.0.0";
	    }
	    else if( header.m_fileVersion == 2 )
	    {
		    return "Havok-3.1.0";
	    }
	    else
	    {
		    HK_ASSERT(0x4d335da9, 0);
		    return HK_NULL;
	    }
	}

	static inline bool IsPublicClassObject(const hkClass* klass)
	{
		const char* klassName = klass->getName();
		return ( hkString::strCmp(klassName, "hkClass") != 0
			&& hkString::strCmp(klassName, "hkClassMember") != 0
			&& hkString::strCmp(klassName, "hkClassEnum") != 0
			&& hkString::strCmp(klassName, "hkClassEnumItem") != 0 );
	}
	
	// 
	// COM-608
	// Find and track all packfile hkClass pointers.
	// Invalidate class registry when tracker is updated (replaceObject()/removeFinish()).
	//
	class hkContentsUpdateTracker : public hkPackfileObjectUpdateTracker
	{
		inline void retrackVariantClassPointers( hkVariant* v, int numVariants )
		{
			for( int i = 0; i < numVariants; ++i )
			{
				if( v[i].m_object && v[i].m_class )
				{
					objectPointedBy(const_cast<hkClass*>(v[i].m_class), &v[i].m_class);
				}
			}
		}

		// find (possibly resurrected) and track hkVariant classes, track homogeneous array classes
		void retrackVariantAndHomogeneousClasses( void* pointer, const hkClass& klass, int numObjs )
		{
			for( int memberIndex = 0; memberIndex < klass.getNumMembers(); ++memberIndex )
			{
				const hkClassMember& member = klass.getMember(memberIndex);
				if( member.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) )
				{
					continue;
				}
				switch( member.getType() )
				{
					case hkClassMember::TYPE_VARIANT:
					{
						int nelem = member.getCstyleArraySize() ? member.getCstyleArraySize() : 1;
						void* obj = pointer;
						int objCount = numObjs;
						while( --objCount >= 0 )
						{
							hkClassMemberAccessor maccess(obj, &member);
							retrackVariantClassPointers( static_cast<hkVariant*>(maccess.asRaw()), nelem );
							obj = hkAddByteOffset(obj, klass.getObjectSize());
						}
						break;
					}
					case hkClassMember::TYPE_ARRAY:
					case hkClassMember::TYPE_SIMPLEARRAY:
					{
						if( member.getSubType() == hkClassMember::TYPE_VARIANT )
						{
							void* obj = pointer;
							int objCount = numObjs;
							while( --objCount >= 0 )
							{
								hkClassMemberAccessor maccess(obj, &member);
								hkClassMemberAccessor::SimpleArray& array = maccess.asSimpleArray();
								retrackVariantClassPointers( static_cast<hkVariant*>(array.data), array.size );
								obj = hkAddByteOffset(obj, klass.getObjectSize());
							}
						}
						else if( member.getSubType() == hkClassMember::TYPE_STRUCT )
						{
							HK_ASSERT(0x556380a8, member.hasClass());
							void* obj = pointer;
							int objCount = numObjs;
							while( --objCount >= 0 )
							{
								hkClassMemberAccessor maccess(obj, &member);
								hkClassMemberAccessor::SimpleArray& array = maccess.asSimpleArray();
								retrackVariantAndHomogeneousClasses( array.data, member.getStructClass(), array.size );
								obj = hkAddByteOffset(obj, klass.getObjectSize());
							}
						}
						break;
					}
					case hkClassMember::TYPE_HOMOGENEOUSARRAY:
					{
						void* obj = pointer;
						int objCount = numObjs;
						while( --objCount >= 0 )
						{
							hkClassMemberAccessor maccess(obj, &member);
							hkClassMemberAccessor::HomogeneousArray& homogeneousArray = maccess.asHomogeneousArray();
							if( homogeneousArray.klass )
							{
								objectPointedBy(homogeneousArray.klass, &homogeneousArray.klass);
								retrackVariantAndHomogeneousClasses( homogeneousArray.data, *homogeneousArray.klass, homogeneousArray.size );
							}
							obj = hkAddByteOffset(obj, klass.getObjectSize());
						}
						break;
					}
					case hkClassMember::TYPE_STRUCT:
					{
						int nelem = member.getCstyleArraySize() ? member.getCstyleArraySize() : 1;
						void* obj = pointer;
						int objCount = numObjs;
						while( --objCount >= 0 )
						{
							hkClassMemberAccessor maccess(obj, &member);
							retrackVariantAndHomogeneousClasses( maccess.asRaw(), member.getStructClass(), nelem );
							obj = hkAddByteOffset(obj, klass.getObjectSize());
						}
						break;
					}
					default:
					{
						// skip over all other types
					}
				}
			}
		}

		public:
			hkContentsUpdateTracker(hkPackfileData* data, hkArray<hkVariant>& objects, hkRefPtr<hkChainedClassNameRegistry>& packfileClassRegistry) :
				hkPackfileObjectUpdateTracker(data), m_dirty(false), m_packfileClassRegistry(packfileClassRegistry)
			{
				HK_ASSERT(0x3a95e389, m_packfileClassRegistry);
				for( int i = 0; i < objects.getSize(); ++i )
				{
					HK_ASSERT(0x5f3fb842, objects[i].m_class);
					retrackVariantAndHomogeneousClasses(objects[i].m_object, *objects[i].m_class, 1);
				}
			}

			inline hkBool32 isDirty()
			{
				return m_dirty;
			}

			inline void setDirty()
			{
				m_dirty = true;
				m_packfileClassRegistry = HK_NULL;
			}

			virtual void replaceObject( void* oldObject, void* newObject, const hkClass* newClass ) // changed/removed
			{
				setDirty();
				hkPackfileObjectUpdateTracker::replaceObject(oldObject, newObject, newClass);
			}

			virtual void removeFinish( void* oldObject ) // changed/renamed/removed
			{
				setDirty();
				hkPackfileObjectUpdateTracker::removeFinish(oldObject);
			}

			hkBool32 m_dirty;
			hkRefPtr<hkChainedClassNameRegistry>& m_packfileClassRegistry;
	};

	class ClassUpdateTracker : public hkObjectUpdateTracker
	{
		public:

			ClassUpdateTracker(hkPackfileData* data) : m_data(data), m_copy(HK_NULL) {}
			virtual void addAllocation(void* p) { HK_ASSERT(0x594e2ebe, 0); m_data->addAllocation(p); }
			virtual void addChunk(void* p, int n, HK_MEMORY_CLASS c) { HK_ASSERT(0x594e2ebe, m_copy == HK_NULL); m_data->addChunk(p,n,c); m_copy = p; m_copySize = n; }
			virtual void objectPointedBy( void* newObject, void* fromWhere )
			{
				void* oldObject = *static_cast<void**>(fromWhere);
				if (oldObject)
				{
					for( int i = m_pointers.getFirstIndex(oldObject);
						i != -1;
						i = m_pointers.getNextIndex(i) )
					{
						if( m_pointers.getValue(i) == fromWhere )
						{
							if( newObject == oldObject )
							{
								return;
							}
							m_pointers.removeByIndex(oldObject, i);
							break;
						}
					}
				}
				if( newObject )
				{
					m_pointers.insert( newObject, fromWhere );
				}
				*static_cast<void**>(fromWhere) = newObject;
			}

			virtual void replaceObject( void* oldObject, void* newObject, const hkClass* newClass )
			{
				// replace pointers to old object with pointers to new one
				int index = m_pointers.getFirstIndex(oldObject);
				if( newObject )
				{
					m_pointers.m_indexMap.insert( newObject, index );
				}
				while( index != -1 )
				{
					void* ptrOldObject = m_pointers.getValue(index);
					HK_ASSERT(0x7fe24edd, *static_cast<void**>(ptrOldObject) == oldObject );
					*static_cast<void**>(ptrOldObject) = newObject;
					index = m_pointers.getNextIndex(index);
				}
			}
			virtual void addFinish( void* newObject, const char* className ) {}
			virtual void removeFinish( void* oldObject ) {}
			hkPackfileData* m_data;
			void* m_copy;
			int m_copySize;
			hkSerializeMultiMap<void*, void*> m_pointers;
	};

	static inline bool mayContainWrongMetadata(const char* contentsVersion)
	{
		// These are wildcards because of the way string matching is done.
		static const char* versionsWithWrongMetadataBug[] =
		{
			"Havok-5.0.0-b1",
			"Havok-5.0.0-r1",
			"Havok-5.1.0-r1",
			HK_NULL
		};
		for( int i = 0; versionsWithWrongMetadataBug[i] != HK_NULL; ++i )
		{
			const char* ver = versionsWithWrongMetadataBug[i];
			if( hkString::strNcmp(contentsVersion, ver, hkString::strLen(ver) ) == 0)
			{
				return true;
			}
		}
		return false;
	}

#if defined(HK_DEBUG)
	// hkClass signature fix. HVK-3680
	static inline bool containsCorrectSignatures(const char* contentsVersion)
	{
		// These are wildcards because of the way string matching is done.
		static const char* versionsWithSignatureBug[] =
		{
			"Havok-3",
			"Havok-4.0",
			"Havok-4.1",
			"Havok-4.5.0-b1",
			"Havok-4.6.0-r1",
			HK_NULL
		};
		for( int i = 0; versionsWithSignatureBug[i] != HK_NULL; ++i )
		{
			const char* ver = versionsWithSignatureBug[i];
			if( hkString::strNcmp(contentsVersion, ver, hkString::strLen(ver) ) == 0)
			{
				return false;
			}
		}
		return true;
	}
#endif

	// COM-434, fix hkClassEnumClass pointers for
	// hkxSparselyAnimatedEnumClass and hctAttributeDescriptionClass
	static void fixClassEnumClassPointers(hkClass& klass)
	{
		if( /*hkString::strCmp(klass.getName(), "hkxSparselyAnimatedEnum") == 0 || */
			hkString::strCmp(klass.getName(), "hctAttributeDescription") == 0 )
		{
			const char* memberNameToFix = /* hkString::strCmp(klass.getName(), "hkxSparselyAnimatedEnum") == 0 ? "type" : */ "enum";
			hkInternalClassMember* m = reinterpret_cast<hkInternalClassMember*>(const_cast<hkClassMember*>(klass.getMemberByName(memberNameToFix)));
			HK_ASSERT(0x45ae0d4a, m);
			if( m->m_class == HK_NULL )
			{
				// fix hkClassEnumClass pointer for
				// hkxSparselyAnimatedEnum::m_type and hctAttributeDescription::m_enum
				HK_ASSERT(0x45ae0d4b, m->m_type == hkClassMember::TYPE_POINTER && m->m_subtype == hkClassMember::TYPE_STRUCT);
				m->m_class = &hkClassEnumClass;
			}
		}
	}

	static inline const hkClassNameRegistry* getGlobalRegistryForPackfileVersion(const char* version)
	{
		if( version == HK_NULL )
		{
			return HK_NULL;
		}
		if( hkString::strCmp(hkVersionUtil::getCurrentVersion(), version) == 0 )
		{
			return hkBuiltinTypeRegistry::getInstance().getClassNameRegistry();
		}

		return hkVersionRegistry::getInstance().getClassNameRegistry(version);
	}

	// COM-608
	static inline void addNonSerializableClasses(hkRefPtr<hkChainedClassNameRegistry>& registry, int fileVersion)
	{
		// add non serialized classes
		const hkClass* specials[] = { &hkClassClass, &hkClassMemberClass, &hkClassEnumClass, &hkClassEnumItemClass };
		if( fileVersion == 1 )
		{
			specials[0] = &hkClassVersion1Class;
			specials[1] = &hkClassMemberVersion1Class;
		}
		else if(fileVersion < 8)
		{
			specials[0] = &hkClassVersion3Class;
			specials[1] = &hkClassMemberVersion3Class;
		}
		for( int i = 0; i < static_cast<int>(HK_COUNT_OF(specials)); ++i )
		{
			registry->registerClass(specials[i]);
		}
	}

	// COM-608
	static inline void validateAndRegisterClasses(hkDynamicClassNameRegistry& registry, const hkClass* topClass)
	{
		if( topClass == HK_NULL || registry.getClassByName(topClass->getName()) )
		{
			return;
		}
		registry.registerClass(topClass);
		validateAndRegisterClasses(registry, topClass->getParent());
		for( int i = 0; i < topClass->getNumDeclaredMembers(); ++i )
		{
			const hkClassMember& m = topClass->getDeclaredMember(i);
			validateAndRegisterClasses(registry, m.getClass());
		}
	}

	// COM-608
	static void findAndRegisterHomogeneousClassesIn(hkDynamicClassNameRegistry& registry, const void* pointer, const hkClass& c, int numObjs)
	{
		for( int i = 0; i < c.getNumDeclaredMembers(); ++i )
		{
			const hkClassMember& m = c.getDeclaredMember(i);
			if( m.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) )
			{
				continue;
			}
			switch( m.getType() )
			{
				case hkClassMember::TYPE_STRUCT:
				{
					HK_ASSERT(0x26bfa446, m.hasClass());
					int nelem = m.getCstyleArraySize() ? m.getCstyleArraySize() : 1;
					const void* o = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor structMember(const_cast<void*>(o), &m);
						findAndRegisterHomogeneousClassesIn(registry, structMember.asRaw(), m.getStructClass(), nelem);
						o = hkAddByteOffsetConst(o, c.getObjectSize());
					}
					break;
				}
				case hkClassMember::TYPE_ARRAY:
				case hkClassMember::TYPE_SIMPLEARRAY:
				{
					if( m.getSubType() == hkClassMember::TYPE_STRUCT )
					{
						HK_ASSERT(0x5fc65b7b, m.hasClass());
						const void* o = pointer;
						int objCount = numObjs;
						while( --objCount >= 0 )
						{
							hkClassMemberAccessor arrayMember(const_cast<void*>(o), &m);
							hkClassMemberAccessor::SimpleArray& sa = arrayMember.asSimpleArray();
							if( sa.data && sa.size )
							{
								findAndRegisterHomogeneousClassesIn(registry, sa.data, m.getStructClass(), sa.size);
							}
							o = hkAddByteOffsetConst(o, c.getObjectSize());
						}
					}
					break;
				}
				case hkClassMember::TYPE_HOMOGENEOUSARRAY:
				{
					const void* o = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor arrayMember(const_cast<void*>(o), &m);
						hkClassMemberAccessor::HomogeneousArray& ha = arrayMember.asHomogeneousArray();
						validateAndRegisterClasses(registry, ha.klass);
						if( ha.klass && ha.data && ha.size )
						{
							findAndRegisterHomogeneousClassesIn(registry, ha.data, *ha.klass, ha.size);
						}
						o = hkAddByteOffsetConst(o, c.getObjectSize());
					}
					break;
				}
				default:
				{
					// skip over all other types
				}
			}
		}
	}
}

hkBinaryPackfileReader::hkBinaryPackfileReader()
:	m_header(HK_NULL),
	m_streamOffset(0),
	m_loadedObjects(HK_NULL),
	m_tracker(HK_NULL),
	m_packfileClassRegistry(HK_NULL)
{
	m_packfileData = new BinaryPackfileData();
	const char* lastVer = hkVersionUtil::getDeprecatedCurrentVersion();
	m_packfileData->setPackfileClassNameRegistry( hkVersionRegistry::getInstance().getClassNameRegistry(lastVer) );
}

hkBinaryPackfileReader::~hkBinaryPackfileReader()
{
	delete m_tracker;
	delete m_loadedObjects;
	m_packfileData->removeReference();
}

hkPackfileData* hkBinaryPackfileReader::getPackfileData() const
{
	TRACE_FUNC_ENTER(">> getPackfileData:");
	HK_ASSERT2(0x24e3df45, isVersionUpToDate(),
		"Make sure the versioning is completed before accessing the packfile data.");

	// check signatures, fix hkVariant class pointers
	getLoadedObjects();

	if( !m_packfileData->finishedObjects() )
	{
		bool useTemporaryTracker = m_tracker == HK_NULL;
		getUpdateTracker(); // make sure we track all the loaded objects
		if( const hkClassNameRegistry* globalRegistry = getGlobalRegistryForPackfileVersion(getContentsVersion()) )
		{
			useClassesFromRegistry(*globalRegistry);
		}
		// We need to use the tracker to get the latest location of all objects
		typedef hkPointerMap<void*, const char*> Map;
		Map& pmap = m_tracker->m_finish;
		for( Map::Iterator it = pmap.getIterator(); pmap.isValid(it); it = pmap.getNext(it) )
		{
			void* objPtr = pmap.getKey(it);
			const char* className = pmap.getValue(it);
			m_packfileData->trackObject(objPtr, className);
		}

		TRACE("Set packfile data class registry.");
		m_packfileData->setPackfileClassNameRegistry(getClassNameRegistry());
		m_packfileData->setContentsWithName(m_tracker->m_topLevelObject, getContentsClassName());
		if( useTemporaryTracker )
		{
			HK_ASSERT(0x34d650dc, m_tracker->getReferenceCount() == 1);
			m_tracker->removeReference();
			m_tracker = HK_NULL;
		}
	}

	TRACE_FUNC_EXIT("<< getPackfileData.");
	return m_packfileData;
}

hkResult hkBinaryPackfileReader::loadEntireFile(hkStreamReader* reader)
{
	HK_ASSERT2( 0x7d8aae87, m_packfileData->isDirty() == false, "You must use new reader for each time you load a packfile.");

	if( loadFileHeader(reader) == HK_SUCCESS )
	{
		// return HK_FAILURE if the packfile is for the wrong platform
		if ( ( m_header->m_layoutRules[0] == hkStructureLayout::HostLayoutRules.m_bytesInPointer ) &&
			 ( m_header->m_layoutRules[1] == hkStructureLayout::HostLayoutRules.m_littleEndian ) &&
			 ( m_header->m_layoutRules[2] == hkStructureLayout::HostLayoutRules.m_reusePaddingOptimization ) &&
			 ( m_header->m_layoutRules[3] == hkStructureLayout::HostLayoutRules.m_emptyBaseClassOptimization ) )
		{
			HK_SERIALIZE_LOG(("\nTrace(func='hkBinaryPackfileReader::loadEntireFile()', fileversion=%d,contentsversion='%s')\n", m_header->m_fileVersion, ::getOriginalContentsVersion(*m_header)));
			if( loadSectionHeadersNoSeek(reader) == HK_SUCCESS )
			{
				for( int i = 0; i < m_header->m_numSections; ++i )
				{
					if( loadSectionNoSeek(reader, i) == HK_FAILURE )
					{
						return HK_FAILURE;
					}
				}
				if( fixupGlobalReferences() == HK_FAILURE )
				{
					return HK_FAILURE;
				}
				return HK_SUCCESS;
			}
		}
		else
		{
			HK_WARN( 0x517ab45c, "Skipping binary packfile from a different platform." );
		}
	}
	return HK_FAILURE;
}

hkResult hkBinaryPackfileReader::loadEntireFileInplace( void* data, int dataSize )
{
	HK_ASSERT2( 0x78b7ad37, dataSize > hkSizeOf(hkPackfileHeader), "Packfile is too small" );
	HK_ASSERT2( 0x7d8aae85, (hkUlong(data) & 0xf) == 0, "Data needs to be 16 byte aligned");
	HK_ASSERT2( 0x7d8aae86, m_packfileData->isDirty() == false, "You must use new reader for each time you load a packfile.");

	// Header
	hkPackfileHeader magic;
	hkPackfileHeader* header = (hkPackfileHeader*)data;
	if( (header->m_magic[0] == magic.m_magic[0])
		&& (header->m_magic[1] == magic.m_magic[1]) )
	{
		m_header = header;
		setContentsVersion( ::getOriginalContentsVersion(*m_header) );
	}
	else
	{
		HK_ASSERT2(0xea701934, 0, "Unable to identify binary inplace data. Is this from a binary file? " );
		return HK_FAILURE;
	}
	HK_ASSERT2(0xfe567fe4, m_header->m_layoutRules[0] == hkStructureLayout::HostLayoutRules.m_bytesInPointer, "Trying to load a binary file with a different pointer size than this platform." );
	HK_ASSERT2(0xfe567fe5, m_header->m_layoutRules[1] == hkStructureLayout::HostLayoutRules.m_littleEndian, "Trying to load a binary file with a different endian than this platform." );
	HK_ASSERT2(0xfe567fe6, m_header->m_layoutRules[2] == hkStructureLayout::HostLayoutRules.m_reusePaddingOptimization, "Trying to load a binary file with a different padding optimization than this platform." );
	HK_ASSERT2(0xfe567fe7, m_header->m_layoutRules[3] == hkStructureLayout::HostLayoutRules.m_emptyBaseClassOptimization, "Trying to load a binary file with a different empty base class optimization than this platform." );

	if(m_header->m_numSections > 0)
	{
		// The size of the hkPackfileSectionHeader changed in version 11 of the binary packfile format.
		// While now it is 64 bytes, before it was 48. We need to be able to read both formats.
		// Because the changing fields are just padding at the end of the header, we can simply reinterpret cast the
		// memory.
		int sectionHeaderSize;
		if(m_header->m_fileVersion <= 10)
		{
			sectionHeaderSize = 12*sizeof(hkInt32);
		}
		else if(m_header->m_fileVersion == 11)
		{
			sectionHeaderSize = 16*sizeof(hkInt32);
		}
		else
		{
			HK_ASSERT2(0xbd3d76c, false, "Invalid handling of new packfile version, or packfile data corrupted.");
			sectionHeaderSize = 0;
		}

		// section headers
		m_sections.init(header+1, sectionHeaderSize);

		if( m_header->m_fileVersion < 4 )
		{
			updateSectionHeaders();
		}
	}

	// section data
	char* dataPtr = reinterpret_cast<char*>( data );
	m_sectionData.setSize(m_header->m_numSections);
	for( int sectionIndex = 0; sectionIndex < m_header->m_numSections; ++sectionIndex )
	{
		hkPackfileSectionHeader& section = m_sections[sectionIndex];

		int sectStart = section.m_absoluteDataStart;
		HK_ON_DEBUG( int sectSize = section.m_endOffset );
		HK_ASSERT2(0xff668fe7, ( (sectStart <= dataSize) && ((sectStart + sectSize) <= dataSize) ), "Inplace packfile data is too small. Is it corrupt?");

		// apply local fixups now
		char* dataBegin = dataPtr + sectStart;
		int* localFixups = reinterpret_cast<int*>(dataBegin + section.m_localFixupsOffset);
		for( int i = 0; i < section.getLocalSize() / hkSizeOf(hkInt32); i+=2 )
		{
			int srcOff = localFixups[i  ];
			if( srcOff == -1 ) continue;
			HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
			int dstOff = localFixups[i+1];
			void** addrSrc = reinterpret_cast<void**>(dataBegin+srcOff);
			void* dst = reinterpret_cast<void*>(dataBegin+dstOff);
			HK_ASSERT2( 0x75936f92, *addrSrc == HK_NULL,
				"Pointer has already been patched. Corrupt file or loadEntireFileInplace called multiple times?");
			*addrSrc = dst;
		}

		m_sectionData[sectionIndex] = dataBegin;
	}

	// todo: merge this with other block in loadEntireFile
	{
		int nameIndex = m_header->m_contentsClassNameSectionIndex;
		int nameOffset = m_header->m_contentsClassNameSectionOffset;

		if( nameIndex >= 0
			&& nameOffset >= 0
			&& m_header->m_fileVersion < 3)
		{
			// contentsClass used to point to a class instance
			// adjust it to point to the class name instead.
			const hkClass* klass = (const hkClass*)getSectionDataByIndex(nameIndex,nameOffset);
			const char* name = klass->getName();
			hkUlong off = hkUlong(name) - hkUlong( m_sectionData[nameIndex] );
			m_header->m_contentsClassNameSectionOffset = int(off);
		}
	}

	return fixupGlobalReferences();
}

hkObjectUpdateTracker& hkBinaryPackfileReader::getUpdateTracker() const
{
	if( m_tracker == HK_NULL )
	{
		TRACE_FUNC_ENTER(">> getUpdateTracker:");
		HK_SERIALIZE_LOG(("\nTrace(func='hkBinaryPackfileReader::getUpdateTracker()')\n"));
		// lazily create tracker if needed
		hkRefPtr<const hkClassNameRegistry> classReg = getClassNameRegistry(); // setup classes

		// track objects and also update packfile class registry
		m_tracker = new hkContentsUpdateTracker(m_packfileData, getLoadedObjects(), m_packfileClassRegistry);
		hkBool shouldWalkPointers = m_header->m_fileVersion == 1;

		//PRINT(("\n### BUILD TRACKER\n"));

		for( int sectionIndex = 0; sectionIndex < m_header->m_numSections; ++sectionIndex )
		{
			if( m_sectionData[sectionIndex] ) // is section loaded?
			{
				const hkPackfileSectionHeader& sect = m_sections[sectionIndex];
				char* dataBegin = static_cast<char*>(m_sectionData[sectionIndex]);
				if( shouldWalkPointers == false )
				{
					int* globalFixups = reinterpret_cast<int*>(dataBegin + sect.m_globalFixupsOffset );
					for( int i = 0; i < sect.getGlobalSize() / hkSizeOf(hkInt32); i += 3 )
					{
						int srcOff = globalFixups[i  ];
						if( srcOff == -1 ) continue;
						HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
						int dstSec = globalFixups[i+1];
						int dstOff = globalFixups[i+2];

						// automatically checks for dest section loaded
						void* dstPtr = getSectionDataByIndex(dstSec, dstOff);
						m_tracker->objectPointedBy( dstPtr, dataBegin+srcOff );
						//PRINT(("GFIX\t0x%p\tat\t0x%p\n", dstPtr, (void*)(dataBegin+srcOff)));
					}
				}
				{
					int* finishFixups = reinterpret_cast<int*>(dataBegin + sect.m_virtualFixupsOffset);
					for( int i = 0; i < sect.getFinishSize() / hkSizeOf(hkInt32); i += 3 )
					{
						int srcOff = finishFixups[i  ];
						if( srcOff == -1 ) continue;
						HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
						int dstSec = finishFixups[i+1];
						int dstOff = finishFixups[i+2];

						void* srcPtr = getSectionDataByIndex(sectionIndex, srcOff);
						void* dstPtr = getSectionDataByIndex(dstSec, dstOff);
						m_tracker->addFinish( srcPtr, static_cast<char*>(dstPtr) );
					}
				}
			}
		}

		m_tracker->setTopLevelObject( getOriginalContents(), getOriginalContentsClassName());

		if( shouldWalkPointers )
		{
			PackfilePointersMapListener listener( *m_tracker, *classReg, m_header->m_fileVersion, getContentsVersion(), m_updateFlagFromClass );
			const hkClass& originalContentsClass = *classReg->getClassByName(getOriginalContentsClassName());
			updateMetaDataInplace(const_cast<hkClass*>(&originalContentsClass), m_header->m_fileVersion, getContentsVersion(), m_updateFlagFromClass);
			//PRINT(("\nTRACK\t0x%p\t# %s(0x%x) at 0x%p\n\n", m_tracker->m_topLevelObject, originalContentsClass.getName(), originalContentsClass.getSignature(), &originalContentsClass));
			if( hkObjectInspector::walkPointers(m_tracker->m_topLevelObject, originalContentsClass, &listener) == HK_FAILURE )
			{
				HK_WARN(0x5a65ee8e, "Error occured while getting the loaded object list from packfile.");
			}
		}
		// fix alignment of c-string pointers, must be at least 2-byte aligned to support hkStringPtr
		if( m_header->m_fileVersion <= 6 )
		{
			PackfileCstringListener listener( *m_tracker, *classReg );
			const hkClass& originalContentsClass = *classReg->getClassByName(getOriginalContentsClassName());
			//PRINT(("\nTRACK\t0x%p\t# %s(0x%x) at 0x%p\n\n", m_tracker->m_topLevelObject, originalContentsClass.getName(), originalContentsClass.getSignature(), &originalContentsClass));
			if( hkObjectInspector::walkPointers(m_tracker->m_topLevelObject, originalContentsClass, &listener) == HK_FAILURE )
			{
				HK_WARN(0x419f5030, "Error occured while checking c-string alignment in packfile.");
			}
		}
		TRACE_FUNC_EXIT("<< getUpdateTracker.");
	}
	return *m_tracker;
}

void hkBinaryPackfileReader::findClassLocations( hkArray<PackfileObject>& classPtrsOut ) const
{
	if( (m_header->m_fileVersion >= 4) || (hkString::strCmp(getContentsVersion(),"Havok-4.0.0-b1")==0) )
	{
		// 400b1 and above have finish for all objects - we can reconstruct from this
		for( int sectionIndex = 0; sectionIndex < m_header->m_numSections; ++sectionIndex )
		{
			if( m_sectionData[sectionIndex] ) // is section loaded?
			{
				const hkPackfileSectionHeader& sect = m_sections[sectionIndex];
				char* dataBegin = static_cast<char*>(m_sectionData[sectionIndex]);
				int* finishFixups = reinterpret_cast<int*>(dataBegin + sect.m_virtualFixupsOffset );
				for( int i = 0; i < sect.getFinishSize() / hkSizeOf(hkInt32); i += 3 )
				{
					int srcOff = finishFixups[i  ];
					if( srcOff == -1 ) continue;
					HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
					int dstSec = finishFixups[i+1];
					int dstOff = finishFixups[i+2];

					const char* className = (const char*)( getSectionDataByIndex(dstSec, dstOff) );
					if( hkString::strCmp(className, "hkClass") == 0 )
					{
						hkClass* klass = reinterpret_cast<hkClass*>(dataBegin+srcOff);
						HK_ASSERT( 0x15a0905a, klass != HK_NULL );
						PackfileObject& p = classPtrsOut.expandOne();
						p.object = klass;
						p.section = dstSec;
						p.offset = dstOff;
					}
				}
			}
		}
	}
	else
	{
		// pre 400 had a special class index section because not all objects had finishing
		int classIndex = getSectionIndex("__classindex__");
		// section exists && non empty && loaded
		if( classIndex >= 0 && m_sections[classIndex].getDataSize() != 0 && m_sectionData[classIndex] != 0)
		{
			hkInt32* data = static_cast<hkInt32*>( m_sectionData[classIndex] );
			int maxEntries = m_sections[classIndex].getDataSize() / hkSizeOf(hkInt32);
			for( int i = 0; i < maxEntries; i += 2 )
			{
				int sec = data[i+0];
				if( sec == -1 ) break;
				int off = data[i+1];

				void* rawClass = getSectionDataByIndex(sec, off);
				if( rawClass )
				{
 					PackfileObject& p = classPtrsOut.expandOne();
					p.object = rawClass;
					p.section = sec;
					p.offset = off;
				}
			}
		}
	}
	for( int i = 0; i < classPtrsOut.getSize(); ++i )
	{
		hkClass* c = static_cast<hkClass*>(classPtrsOut[i].object);
		fixClassEnumClassPointers(*c);
	}
}

const hkClassNameRegistry* hkBinaryPackfileReader::getClassNameRegistry() const
{
	if( m_packfileClassRegistry )
	{
		return m_packfileClassRegistry;
	}
	TRACE_FUNC_ENTER(">> getClassNameRegistry:");
	const hkClassNameRegistry* nextRegistry = getGlobalRegistryForPackfileVersion(getContentsVersion());

	if(!nextRegistry)
	{
		return HK_NULL;
	}

	m_packfileClassRegistry = new hkChainedClassNameRegistry(HK_NULL);
	m_packfileClassRegistry->removeReference(); // hkRefPtr takes responsibility

	if( m_tracker && static_cast<hkContentsUpdateTracker*>(m_tracker)->isDirty() )
	{
		TRACE("using tracker.");
		m_packfileClassRegistry->setNextRegistry(HK_NULL);
		hkArray<hkVariant>& loadedObject = getLoadedObjects();
		for( int i = 0; i < loadedObject.getSize(); ++i )
		{
			hkClass* klass = const_cast<hkClass*>(loadedObject[i].m_class);
			HK_ASSERT(0x3dd5cee7, klass);
			validateAndRegisterClasses(*m_packfileClassRegistry, klass);
			findAndRegisterHomogeneousClassesIn(*m_packfileClassRegistry, loadedObject[i].m_object, *klass, 1);
		}
	}
	else
	{
		TRACE("no tracker.");
		hkArray<PackfileObject> classPointers;
		findClassLocations( classPointers );
		if( classPointers.getSize() > 0 )
		{
			for( int i = 0; i < classPointers.getSize(); ++i )
			{
				hkClass* c = static_cast<hkClass*>(classPointers[i].object);
				hkPackfileReader::updateMetaDataInplace( c, m_header->m_fileVersion, getContentsVersion(), m_updateFlagFromClass );
				m_packfileClassRegistry->registerClass( c, c->getName() );
			}

			// add non serialized classes
			addNonSerializableClasses(m_packfileClassRegistry, m_header->m_fileVersion);
		}
	}
	m_packfileClassRegistry->setNextRegistry(nextRegistry);
	TRACE_FUNC_EXIT("<< getClassNameRegistry.");

	return m_packfileClassRegistry;
}

hkArray<hkVariant>& hkBinaryPackfileReader::getLoadedObjects() const
{
	if( m_loadedObjects == HK_NULL )
	{
		TRACE_FUNC_ENTER(">> getLoadedObjects (" << m_header->m_fileVersion << ", " << getContentsVersion() << "):");
		HK_SERIALIZE_LOG(("\nTrace(func=\"hkBinaryPackfileReader::getLoadedObjects\")\n"));
		const hkClassNameRegistry* classReg = getClassNameRegistry();
		HK_COMPILE_TIME_ASSERT( sizeof(void*) <= 8 );

		// use finish fixups to find exact type of all objects
		// older files may have only virtual objects in these fixups
		//PRINT(("\n### LOAD OBJs\n"));
#		if defined(HK_DEBUG)
			// hkClass signature fix. HVK-3680
			bool hasCorrectSig = m_header->m_fileVersion >= 4 && containsCorrectSignatures(getContentsVersion());
#		endif

		int metadataSectionIndex = -1;
		if( mayContainWrongMetadata(getContentsVersion()) )
		{
			metadataSectionIndex = getSectionIndex("__types__"); // fileversion >=5 only
		}

		ClassFromAddressMap classFromObjectMap;
		for( int sectionIndex = 0; sectionIndex < m_header->m_numSections; ++sectionIndex )
		{
			if( m_sectionData[sectionIndex] ) // is section loaded?
			{
				const hkPackfileSectionHeader& sect = m_sections[sectionIndex];
				char* dataBegin = static_cast<char*>(m_sectionData[sectionIndex]);
				int* finishFixups = reinterpret_cast<int*>(dataBegin + sect.m_virtualFixupsOffset );
				for( int i = 0; i < sect.getFinishSize() / hkSizeOf(hkInt32); i += 3 )
				{
					int srcOff = finishFixups[i  ];
					if( srcOff == -1 ) continue;
					HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
					int dstSec = finishFixups[i+1];
					int dstOff = finishFixups[i+2];

					// If __classnames__ has signatures, check them against the used registry.
					char* className = static_cast<char*>( getSectionDataByIndex(dstSec, dstOff) );
					const hkClass* klass = classReg->getClassByName(className);

					HK_ASSERT3( 0x15a09058, klass != HK_NULL, "Found an unregistered class " << className );
					// check if it is not the runtime class
					if( metadataSectionIndex != i &&
						IsPublicClassObject(klass) )
					{
#						if defined(HK_DEBUG)
							if( dstOff != 0 && className[-1] == '\t' && hasCorrectSig /* hkClass signature fix. HVK-3680 */ ) // signature exists
							{
								union { char c[4]; hkUint32 sig; } u;
								u.c[0] = className[-5];
								u.c[1] = className[-4];
								u.c[2] = className[-3];
								u.c[3] = className[-2];
								HK_ON_DEBUG(hkUlong sig = klass->getSignature());
								HK_ASSERT3(0x15a09059, hkUlong(sig) == u.sig, "Signature mismatch " << className << " (0x" << klass << ") - 0x" << (void*)sig << ", 0x" << (void*)((hkUlong)u.sig));
							}
#						endif
						classFromObjectMap.insert(static_cast<void*>(dataBegin+srcOff), const_cast<hkClass*>(klass));
						HK_SERIALIZE_LOG(("FinishFixup(addr=0x%p,klass=\"%s\",vtable=%i)\n", static_cast<void*>(dataBegin+srcOff), klass->getName(), klass->hasVtable() ? 1 : 0));
					}
				}
			}
		}
		// lazily create loaded objects
		m_loadedObjects = new hkArray<hkVariant>();
		PackfileObjectsCollector collector( *m_loadedObjects, classFromObjectMap, m_header->m_fileVersion, getContentsVersion(), m_updateFlagFromClass );
		// reserve memory for all known objects
		m_loadedObjects->reserve(classFromObjectMap.getSize());
		const hkClass& originalContentsClass = *classReg->getClassByName(getOriginalContentsClassName());
		if( hkObjectInspector::walkPointers(getOriginalContents(), originalContentsClass, &collector) == HK_FAILURE )
		{
			HK_WARN(0x5a65ee8e, "Error occured while getting the loaded object list from packfile.");
		}
		TRACE_FUNC_EXIT("<< getLoadedObjects.");
	}
	return *m_loadedObjects;
}

hkVariant hkBinaryPackfileReader::getTopLevelObject() const
{
	hkVariant topLevelObjectVariant = { m_tracker ? m_tracker->m_topLevelObject : getOriginalContents(), HK_NULL };
	if( const hkClassNameRegistry* registry = getGlobalRegistryForPackfileVersion(getContentsVersion()) )
	{
		useClassesFromRegistry(*registry);
	}
	topLevelObjectVariant.m_class = getClassNameRegistry()->getClassByName(getContentsClassName());
	return topLevelObjectVariant;
}

void* hkBinaryPackfileReader::getContentsWithRegistry(
	const char* expectedClassName,
	const hkTypeInfoRegistry* finishRegistry)
{
	TRACE_FUNC_ENTER(">> getContentsWithRegistry:");
	if( finishRegistry != HK_NULL )
	{
		warnIfNotUpToDate();
	}

	// Get toplevel
	const char* topLevelClassName = getContentsClassName();

	// Check contents are what we expect
	if( expectedClassName != HK_NULL && topLevelClassName != HK_NULL )
	{
		const hkClassNameRegistry* classReg = getClassNameRegistry();
		HK_ASSERT(0x31224ab7, classReg);
		const hkClass* expectedClass = classReg->getClassByName(expectedClassName);
		HK_ASSERT(0x4eff213b, expectedClass);
		const hkClass* topLevelClass = classReg->getClassByName(topLevelClassName);
		HK_ASSERT(0x63ed5156, topLevelClass);
		if( !expectedClass->isSuperClass(*topLevelClass) )
		{
			HK_WARN(0x599a0b0c, "Requested " << expectedClassName << " but file contains " << topLevelClassName);
			return HK_NULL;
		}
	}

	// check signatures, fix hkVariant class pointers
	getLoadedObjects();

	if( finishRegistry != HK_NULL )
	{
		// Fixup vtables, finish objects, etc.
		if( finishLoadedObjects(*finishRegistry) == HK_FAILURE )
		{
			return HK_NULL;
		}
	}
	else
	{
		// substitute hkClass* with built-in hkClass* where possible
		if( const hkClassNameRegistry* registry = getGlobalRegistryForPackfileVersion(getContentsVersion()) )
		{
			useClassesFromRegistry(*registry);
		}

		TRACE("Set packfile data class registry.");
		m_packfileData->setPackfileClassNameRegistry(getClassNameRegistry());
	}

	TRACE_FUNC_EXIT("<< getContentsWithRegistry...");
	return m_packfileData->getContentsPointer(topLevelClassName, finishRegistry);
}

hkResult hkBinaryPackfileReader::loadFileHeader(hkStreamReader* reader, hkPackfileHeader* dst )
{
	HK_ASSERT( 0x78b8ad37, reader );
	// invalidate class registry
	m_packfileClassRegistry = HK_NULL;
	m_packfileData->setPackfileClassNameRegistry(HK_NULL);

	// Store starting point for Havok data... will use this for all reads
	m_streamOffset = 0;
	if( hkSeekableStreamReader* sr = reader->isSeekTellSupported() )
	{
		m_streamOffset = sr->tell();
	}

	if (!dst)
	{
		dst = hkAllocate<hkPackfileHeader>( 1, HK_MEMORY_CLASS_EXPORT  );
		m_packfileData->addAllocation(dst);
	}

	if( reader->read( dst, sizeof(hkPackfileHeader)) == sizeof(hkPackfileHeader) )
	{
		hkPackfileHeader* header = (hkPackfileHeader*)dst;
		hkPackfileHeader magic;
		if( (header->m_magic[0] == magic.m_magic[0])
			&& (header->m_magic[1] == magic.m_magic[1]) )
		{
			m_header = dst;
			setContentsVersion( ::getOriginalContentsVersion(*m_header) );
			return HK_SUCCESS;
		}
	}

	m_header = HK_NULL;
	HK_ASSERT2(0xea701934, 0, "Unable to identify binary file header. Is this a binary file? " );
	return HK_FAILURE;
}

const hkPackfileHeader& hkBinaryPackfileReader::getFileHeader() const
{
	HK_ASSERT2(0xea701935, m_header, "You must load the file header or an entire file first on a proper stream.");
	return *m_header;
}

int hkBinaryPackfileReader::getNumSections() const
{
	return m_header->m_numSections;
}

hkResult hkBinaryPackfileReader::loadSectionHeadersNoSeek( hkStreamReader* reader, hkPackfileSectionHeader* dst )
{
	// invalidate class registry
	m_packfileClassRegistry = HK_NULL;
	m_packfileData->setPackfileClassNameRegistry(HK_NULL);

	if( dst == HK_NULL )
	{
		dst = hkAllocate<hkPackfileSectionHeader>( m_header->m_numSections, HK_MEMORY_CLASS_EXPORT );
		m_packfileData->addAllocation(dst);
	}

	// The size of the hkPackfileSectionHeader changed in version 11 of the binary packfile format.
	// While now it is 64 bytes, before it was 48. We need to be able to read both formats.
	// Because the changing fields are just padding at the end of the header, we can simply reinterpret cast the
	// memory.
	int sectionHeaderSize;
	if(m_header->m_fileVersion <= 10)
	{
		sectionHeaderSize = 12*sizeof(hkInt32);
	}
	else if(m_header->m_fileVersion == 11)
	{
		sectionHeaderSize = 16*sizeof(hkInt32);
	}
	else
	{
		HK_ASSERT2(0x2c834405, false, "Invalid handling of new packfile version, or packfile data corrupted.");
		sectionHeaderSize = 0;
	}

	int size = sectionHeaderSize * m_header->m_numSections;
	HK_ASSERT(0x728eeb3a, reader);
	if( reader->read( dst, size ) == size )
	{
		m_sections.init(dst, sectionHeaderSize);
		m_sectionData.setSize( m_header->m_numSections, 0 );
		if( m_header->m_fileVersion < 4 )
		{
			updateSectionHeaders();
		}
		return HK_SUCCESS;
	}
	return HK_FAILURE;
}

const hkPackfileSectionHeader& hkBinaryPackfileReader::getSectionHeader(int idx) const
{
	HK_ASSERT(0x3c0dde36, m_sections.isValid());
	return m_sections[idx];
}

hkResult hkBinaryPackfileReader::loadSectionNoSeek(hkStreamReader* reader, int sectionIndex, void* buf)
{
	HK_ASSERT2(0x1c8e05da, m_tracker == HK_NULL, "Can't do versioning and load/unload sections");
	HK_ASSERT2(0x1c8e05db, !m_packfileData->finishedObjects(), "Can't load section when loaded objects are already finished.");

	// invalidate class registry
	m_packfileClassRegistry = HK_NULL;
	m_packfileData->setPackfileClassNameRegistry(HK_NULL);

	hkPackfileSectionHeader& section = m_sections[sectionIndex];

	int sectsize = section.m_endOffset;
	if( buf == HK_NULL )
	{
		buf = hkAllocate<char>( sectsize, HK_MEMORY_CLASS_EXPORT );
		m_packfileData->addAllocation(buf);
	}

	if( reader->read(buf, sectsize ) == sectsize )
	{
		// apply local fixups now
		char* dataBegin = static_cast<char*>(buf);
		int* localFixups = reinterpret_cast<int*>(dataBegin + section.m_localFixupsOffset);
		for( int i = 0; i < section.getLocalSize() / hkSizeOf(hkInt32); i+=2 )
		{
			int srcOff = localFixups[i  ];
			if( srcOff == -1 ) continue;
			HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
			int dstOff = localFixups[i+1];

			*(hkUlong*)(dataBegin+srcOff) = hkUlong(dataBegin+dstOff);
		}

		// exports
		{
			hkArray<hkResource::Export> exports;
			section.getExports(buf, exports);
			for( int i = 0; i < exports.getSize(); ++i )
			{
				m_packfileData->addExport( exports[i].name, exports[i].data );
			}
		}
		// imports
		{
			hkArray<hkResource::Import> imports;
			section.getImports(buf, imports);
			for( int i = 0; i < imports.getSize(); ++i )
			{
				m_packfileData->addImport( imports[i].name, imports[i].location );
			}
		}

		m_sectionData[sectionIndex] = buf;

		int nameIndex = m_header->m_contentsClassNameSectionIndex;
		int nameOffset = m_header->m_contentsClassNameSectionOffset;

		if( sectionIndex == nameIndex
			&& nameOffset >= 0
			&& m_header->m_fileVersion < 3)
		{
			// contentsClass used to point to a class instance
			// adjust it to point to the class name instead.
			const hkClass* klass = (const hkClass*)getSectionDataByIndex(nameIndex,nameOffset);
			const char* name = klass->getName();
			hkUlong off = hkUlong(name) - hkUlong(buf);
			m_header->m_contentsClassNameSectionOffset = int(off);
		}
		return HK_SUCCESS;
	}
	return HK_FAILURE;
}

hkResult hkBinaryPackfileReader::loadSection(hkSeekableStreamReader* reader, int sectionIndex, void* buf)
{
	HK_ASSERT2(0x5b2c4574, reader != HK_NULL, "Read from a closed file." );

	if( reader->seek( m_sections[sectionIndex].m_absoluteDataStart + m_streamOffset, hkSeekableStreamReader::STREAM_SET ) == HK_SUCCESS )
	{
		return loadSectionNoSeek( reader, sectionIndex, buf );
	}
	return HK_FAILURE;
}

void hkBinaryPackfileReader::BinaryPackfileData::freeSection(void* data)
{
	for( int i = 0; i < m_memory.getSize(); ++i )
	{
		// possibly need to deallocate memory
		if( m_memory[i] == data )
		{
			m_memory.removeAt(i);
			hkDeallocate<char>( static_cast<char*>(data));
			break;
		}
	}
}

void hkBinaryPackfileReader::BinaryPackfileData::untrackAndDestructObject( void* obj )
{
	const char* className = m_trackedObjects.getWithDefault(obj, HK_NULL);
	if( className )
	{
		const hkTypeInfo* typeInfo = m_trackedTypes.getWithDefault(className, HK_NULL);
		typeInfo->cleanupLoadedObject(obj);
		if( m_topLevelObject == obj )
		{
			m_topLevelObject = HK_NULL;
		}
		m_trackedObjects.remove(m_trackedObjects.findKey(obj));
	}
}

hkResult hkBinaryPackfileReader::unloadSection(int unloadIndex)
{
	HK_ASSERT2(0x1c8e05da, m_tracker == HK_NULL, "Can't do versioning and load/unload sections");
	hkPackfileSectionHeader& sect = m_sections[unloadIndex];
	char* dataBegin = static_cast<char*>(m_sectionData[unloadIndex]);

	// remove imports
	{
		hkArray<hkResource::Import> imports;
		sect.getImports( dataBegin, imports );
		for( int i = 0; i < imports.getSize(); ++i )
		{
			m_packfileData->removeImport( imports[i].location );
		}
	}

	// remove exports
	{
		hkArray<hkResource::Export> exports;
		sect.getExports( dataBegin, exports );
		for( int i = 0; i < exports.getSize(); ++i )
		{
			m_packfileData->removeExport( exports[i].data );
		}
	}

	// untrack objects
	{
		int* finishFixups = reinterpret_cast<int*>(dataBegin + sect.m_virtualFixupsOffset );
		int finishIndexEnd = findEndFixupIndex( finishFixups, sect.getFinishSize(), 3 );
		for( int finishIndex = 0; finishIndex < finishIndexEnd; finishIndex += 3 )
		{
			int srcOff = finishFixups[finishIndex];
			HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
			void* obj = dataBegin+srcOff;
			m_packfileData->untrackAndDestructObject(obj);
		}
	}

	// invalidate class registry
	m_packfileClassRegistry = HK_NULL;
	m_packfileData->setPackfileClassNameRegistry(HK_NULL);
	// free memory
	m_packfileData->freeSection(m_sectionData[unloadIndex]);
	m_sectionData[unloadIndex] = HK_NULL;

	return HK_SUCCESS;
}

hkResult hkBinaryPackfileReader::fixupGlobalReferences()
{
	TRACE_FUNC_ENTER(">> fixupGlobalReferences: " << m_header->m_fileVersion << ", " << getContentsVersion() << ".");
	for( int sectionIndex = 0; sectionIndex < m_header->m_numSections; ++sectionIndex )
	{
		if( m_sectionData[sectionIndex] ) // is section loaded?
		{
			hkPackfileSectionHeader& sect = m_sections[sectionIndex];
			char* dataBegin = static_cast<char*>(m_sectionData[sectionIndex]);
			int* globalFixups = reinterpret_cast<int*>(dataBegin + sect.m_globalFixupsOffset );
			for( int i = 0; i < sect.getGlobalSize() / hkSizeOf(hkInt32); i += 3 )
			{
				int srcOff = globalFixups[i  ];
				if( srcOff == -1 ) continue;
				HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
				int dstSec = globalFixups[i+1];
				int dstOff = globalFixups[i+2];

				// automatically checks for dest section loaded
				void** srcLoc = reinterpret_cast<void**>(dataBegin+srcOff);
				void* dstPtr = getSectionDataByIndex(dstSec, dstOff);
				HK_SERIALIZE_LOG(("GlobalFixup(obj=0x%p, loc=0x%p)\n", dstPtr, srcLoc));
				*srcLoc = dstPtr;
			}
		}
	}
	if( m_header->m_fileVersion < 5 && m_tracker == HK_NULL )
	{
		if( m_packfileClassRegistry )
		{
			TRACE_FUNC_EXIT("<< fixupGlobalReferences (reuse class registry).");
			return HK_SUCCESS;
		}
		TRACE("no tracker.");
		m_packfileClassRegistry = new hkChainedClassNameRegistry(getGlobalRegistryForPackfileVersion(getContentsVersion()));
		m_packfileClassRegistry->removeReference(); // hkRefPtr takes responsibility

		hkArray<PackfileObject> classLocations;
		findClassLocations(classLocations);
		if( classLocations.getSize() > 0 )
		{
			hkArray<hkVariant> classesInOut;
			hkVariant* v = classesInOut.expandBy(classLocations.getSize());
			for( int i = 0; i < classesInOut.getSize(); ++i )
			{
				v[i].m_object = classLocations[i].object;
				v[i].m_class = &hkClassVersion1Class;
			}
			ClassUpdateTracker tracker(m_packfileData);
			static const hkClass* classes[] =
			{
				&hkClassVersion1PaddedClass,
				HK_NULL
			};
			static const hkStaticClassNameRegistry staticClassReg
			(
				classes,
				-1,
				"internal meta-data versioning"
			);
			static hkVersionRegistry::ClassAction actions[] =
			{
				{ 0, 0, hkVersionRegistry::VERSION_COPY, "hkClass", HK_NULL },
				{ 0, 0, 0, 0, 0 }
			};
			const int classSignatures[] = { 0, 0, 0x0ef90576, 0x0ef90576, 0x0ef90576 };
			if( unsigned(m_header->m_fileVersion) < HK_COUNT_OF(classSignatures) )
			{
				actions[0].oldSignature = classSignatures[ m_header->m_fileVersion ];
			}
			static const hkVersionRegistry::UpdateDescription classUpdateDescription
			(
				HK_NULL, actions, &staticClassReg
			);
			hkVersionUtil::updateSingleVersion( classesInOut, tracker, classUpdateDescription, &staticClassReg );
			{
				int oldNumSections = m_header->m_numSections;
				{
					// If the file version is less than 5 is definitely less than 10.
					// The section header size will be 12*sizeof(hkInt32).
					int sectionHeaderSize = 12*sizeof(hkInt32);
					hkPackfileSectionHeader* newSections = hkAllocate<hkPackfileSectionHeader>( oldNumSections + 1, HK_MEMORY_CLASS_EXPORT );
					m_packfileData->addAllocation( newSections );
					hkString::memCpy( newSections, &m_sections[0], sectionHeaderSize * oldNumSections );
					m_sections.init(newSections, sectionHeaderSize);
				}
				m_header->m_numSections += 1;
				hkPackfileSectionHeader& s = m_sections[oldNumSections];
				hkString::strCpy( s.m_sectionTag, "__types__");
				s.m_nullByte = 0;
				s.m_absoluteDataStart = -1;
				s.m_localFixupsOffset = tracker.m_copySize;
				s.m_globalFixupsOffset = tracker.m_copySize;
				s.m_virtualFixupsOffset = tracker.m_copySize;
				s.m_exportsOffset = tracker.m_copySize;
				s.m_importsOffset = tracker.m_copySize;
				s.m_endOffset = tracker.m_copySize;
			}

			int newTypeIndex = m_sectionData.getSize();
			m_sectionData.pushBack( tracker.m_copy );

			hkPointerMap<void*, int> diversions;
			{
				for( int i = 0; i < classesInOut.getSize(); ++i )
				{
					hkClass* c = static_cast<hkClass*>(classesInOut[i].m_object);
					diversions.insert( classLocations[i].object, i);
					hkPackfileReader::updateMetaDataInplace(c, m_header->m_fileVersion, getContentsVersion(), m_updateFlagFromClass);
					m_packfileClassRegistry->registerClass(c);
				}
			}
			for( int sectionIndex = 0; sectionIndex < m_header->m_numSections; ++sectionIndex )
			{
				HK_ASSERT(0x7ec4e527, m_sectionData[sectionIndex] ); // is section loaded?
				hkPackfileSectionHeader& sect = m_sections[sectionIndex];
				char* dataBegin = static_cast<char*>(m_sectionData[sectionIndex]);
				{
					int* globalFixups = reinterpret_cast<int*>(dataBegin + sect.m_globalFixupsOffset );
					for( int i = 0; i < sect.getGlobalSize() / hkSizeOf(hkInt32); i += 3 )
					{
						int srcOff = globalFixups[i  ];
						if( srcOff == -1 ) continue;
						HK_ASSERT( 0xd207ae6b, (srcOff & (sizeof(void*)-1)) == 0 );
						int dstSec = globalFixups[i+1];
						int dstOff = globalFixups[i+2];
						void* dstPtr = getSectionDataByIndex(dstSec, dstOff);
						int idx = diversions.getWithDefault( dstPtr, -1 );
						if( idx != -1 )
						{
							globalFixups[i+1] = newTypeIndex;
							globalFixups[i+2] = int( hkUlong(classesInOut[idx].m_object) - hkUlong(tracker.m_copy) );
							void** srcLoc = reinterpret_cast<void**>(dataBegin+srcOff);
							*srcLoc = classesInOut[idx].m_object;
						}
					}
				}
			}
		}
	}
	TRACE_FUNC_EXIT("<< fixupGlobalReferences.");
	return HK_SUCCESS;
}

hkResult hkBinaryPackfileReader::finishLoadedObjects(const hkTypeInfoRegistry& finishObjects )
{
	TRACE_FUNC_ENTER(">> finishLoadedObjects:");
	getPackfileData();  // make sure packfile data tracks objects

	if( m_packfileData->getContentsPointer(getContentsClassName(), &finishObjects) )
	{
		TRACE_FUNC_EXIT("<< finishLoadedObjects OK.");
		return HK_SUCCESS;
	}
	TRACE_FUNC_EXIT("<< finishLoadedObjects BAD.");
	return HK_FAILURE;
}

int hkBinaryPackfileReader::getSectionIndex( SectionTag sectionTag ) const
{
	for( int i = 0; i < m_header->m_numSections; ++i )
	{
		if( hkString::strCmp(m_sections[i].m_sectionTag, sectionTag ) == 0 )
		{
			return i;
		}
	}
	return -1;
}

void* hkBinaryPackfileReader::getSectionDataByIndex( int sectionIndex, int offset ) const
{
	if( m_sectionData[sectionIndex] != HK_NULL )
	{
		return static_cast<char*>(m_sectionData[sectionIndex]) + offset;
	}
	return HK_NULL;
}


void* hkBinaryPackfileReader::getOriginalContents() const
{
	int si = m_header->m_contentsSectionIndex;
	int so = m_header->m_contentsSectionOffset;
	// v1 packfiles have m_contentsSection* == -1
	return (si >= 0) && (so >= 0)
		? getSectionDataByIndex(si,so)
		: getSectionDataByIndex(getSectionIndex("__data__"), 0);
}

const char* hkBinaryPackfileReader::getOriginalContentsClassName() const
{
	int si = m_header->m_contentsClassNameSectionIndex;
	int so = m_header->m_contentsClassNameSectionOffset;
	return (si >= 0) && (so >= 0)
		? reinterpret_cast<const char*>(getSectionDataByIndex(si,so))
		: HK_NULL;
}

const char* hkBinaryPackfileReader::getContentsClassName() const
{
	return ( m_tracker != HK_NULL )
		? m_tracker->getTopLevelClassName() // toplevel may have been versioned
		: getOriginalContentsClassName();
}

void hkBinaryPackfileReader::useClassesFromRegistry(const hkClassNameRegistry& registry) const
{
	TRACE_FUNC_ENTER(">> useClassesFromRegistry:");
	bool useTemporaryTracker = m_tracker == HK_NULL;
	hkObjectUpdateTracker& tracker = getUpdateTracker();
	// make sure we track packfile object and class locations
	hkRefPtr<hkChainedClassNameRegistry> classReg = static_cast<hkChainedClassNameRegistry*>(const_cast<hkClassNameRegistry*>(getClassNameRegistry())); // read metadata only, if any
	hkArray<const hkClass*> classes;
	classReg->getClasses(classes);
	for( int i = 0; i < classes.getSize(); ++i )
	{
		const hkClass* klass = classes[i];
		const hkClass* registryClass = registry.getClassByName(klass->getName());
		if( registryClass )
		{
			if( registryClass != klass )
			{
				// special case for hkClassMember and hkClass metadata loaded from old files.
#if defined(HK_DEBUG)
				if((hkString::strCmp(klass->getName(), "hkClassMember") != 0 || klass->getSignature() != 0xb0efa719) && (hkString::strCmp(klass->getName(), "hkClass") != 0 || klass->getSignature() != 0x33d42383))
				{
#	if (HAVOK_BUILD_NUMBER == 0) // Internal builds only!
					
					const char* extraInfo = "\nYou may need to rebuild AssetCc2.exe and re-run it to update this HKX file.";
#	else
					const char* extraInfo = "";
#	endif
					HK_ASSERT3(0x15a0905b, klass->getSignature() == registryClass->getSignature(), "Signature mismatch " << klass->getName() << " (0x" << klass << ") - 0x" << (void*)((hkUlong)klass->getSignature()) << ", 0x" << (void*)((hkUlong)registryClass->getSignature()) << extraInfo);
				}
#endif
				// replace class with version from registry
				classReg->registerClass(registryClass);
				tracker.replaceObject(const_cast<hkClass*>(klass), const_cast<hkClass*>(registryClass), &hkClassClass);
				tracker.removeFinish(const_cast<hkClass*>(registryClass));
				TRACE("Class " << klass->getName() << " is used from the provided registry.");
			}
		}
		else
		{
			HK_WARN(0x38afeb70, "Class " << klass->getName() << " is not found in the provided registry.");
		}
	}
	if( useTemporaryTracker )
	{
		HK_ASSERT(0x482b4147, m_tracker->getReferenceCount() == 1);
		m_tracker->removeReference();
		m_tracker = HK_NULL;
	}
	m_packfileClassRegistry = classReg;
	TRACE_FUNC_EXIT("<< useClassesFromRegistry.");
}

void hkBinaryPackfileReader::updateSectionHeaders()
{
	for( int i = 0; i < m_header->m_numSections; ++i )
	{
		hkPackfileSectionHeader& s = m_sections[i];
		// endOffset moves down a slot in v3 -> v4
		s.m_endOffset = s.m_importsOffset;
		// no imports/exports in v3
		s.m_importsOffset = s.m_endOffset;
		s.m_exportsOffset = s.m_endOffset;
	}
}

hkBinaryPackfileReader::SectionHeaderArray::SectionHeaderArray()
	: m_baseSection(HK_NULL), m_sectionHeaderSize(0)
{}

void hkBinaryPackfileReader::SectionHeaderArray::init(void* baseSection, int sectionHeaderSize)
{
	m_baseSection = baseSection;
	m_sectionHeaderSize = sectionHeaderSize;
}

bool hkBinaryPackfileReader::SectionHeaderArray::isValid() const
{
	return (m_baseSection != HK_NULL);
}

hkPackfileSectionHeader& hkBinaryPackfileReader::SectionHeaderArray::operator[](int i)
{
	HK_ASSERT(0x79d9a695, isValid());
	return *static_cast<hkPackfileSectionHeader*>(hkAddByteOffset(m_baseSection, m_sectionHeaderSize * i));
}

const hkPackfileSectionHeader& hkBinaryPackfileReader::SectionHeaderArray::operator[](int i) const
{
	HK_ASSERT(0x3df93e55, isValid());
	return *static_cast<const hkPackfileSectionHeader*>(hkAddByteOffset(m_baseSection, m_sectionHeaderSize * i));
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
