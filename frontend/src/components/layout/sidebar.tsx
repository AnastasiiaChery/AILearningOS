"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BookOpen,
  Brain,
  FileText,
  LayoutDashboard,
  MessageSquare,
  HelpCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Primary learning flow first; the standalone whole-KB chat is secondary now
// that each plan topic has its own contextual "Ask about this" mentor.
const navItems = [
  { href: "/knowledge", icon: BookOpen, label: "Knowledge Base" },
  { href: "/plans", icon: Brain, label: "Learning Plans" },
  { href: "/quizzes", icon: HelpCircle, label: "Quizzes" },
  { href: "/progress", icon: LayoutDashboard, label: "Progress" },
  { href: "/chat", icon: MessageSquare, label: "Ask (all docs)" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col shrink-0">
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center gap-2">
          <Brain className="w-6 h-6 text-emerald-400" />
          <span className="font-semibold text-sm text-white">AI Learning OS</span>
        </div>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        {navItems.map(({ href, icon: Icon, label }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors",
                active
                  ? "bg-emerald-500/10 text-emerald-400"
                  : "text-gray-400 hover:text-gray-100 hover:bg-gray-800"
              )}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-gray-800 text-xs text-gray-600">
        v0.1.0
      </div>
    </aside>
  );
}
